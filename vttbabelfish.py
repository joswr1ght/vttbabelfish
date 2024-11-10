#!/usr/bin/env python3
import argparse
import os
import logging
import sys
from typing import List, Tuple, Optional
import langcodes
import nltk
import tqdm

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)


class VTTTranslator:
    def __init__(self, api_key: str, platform: str = "anthropic",
                 exclude_terms: str = '', no_progress_bar: bool = False):
        self.platform = platform
        if platform == 'anthropic':
            try:
                from anthropic import Anthropic
            except ImportError:
                logger.error('Anthropic API not found. Please install the "anthropic" package: pip install anthropic')
                sys.exit(1)
            self.client = Anthropic(api_key=api_key)
            self.translate_text = self.translate_text_anthropic
        elif platform == 'chatgpt':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                logger.error('OpenAI API not found. Please install the "openai" package: pip install openai')
                sys.exit(1)
            self.translate_text = self.translate_text_chatgpt

        self.no_progress_bar = no_progress_bar
        self.exclude_terms = exclude_terms
        self.prev_translations = []
        self.sentences = []

        self.user_prompt = """
Translate this sentence into {target_lang_name}.
Ignore the dollar signs in the text, but preserve them and their placement in the translated text.

Sentence to translate in triple quotes: '''{text}'''

Requirements:
1. You MUST translate all regular text into proper, natural {target_lang_name}
2. The translation must preserve the dollar sign delimiters from the original text
3. The same number of delimiters must be present in the translated text
4. When restoring the dollar sign delimiter in translated text, place the dollar sign AFTER any adjacent punctuation
5. Only preserve the following exactly as-is:
- Technical terms (e.g., {exclude_terms})
- Proper nouns for security assessment tools (e.g., Burp Suite, Metasploit, Wireshark, Nmap, Legba, Netcat)
- Commands and code snippets
- Proper names (e.g., Joshua Wright, Jeff McJunkin)
- URLs or file paths

Examples of good translations:
Input: "Hi, I'm Joshua Wright,$ and in this lab we'll$ look at command$ injection attacks."
Output: "Hola, soy Joshua Wright,$ y en este laboratorio veremos$ los ataques de Command Injection$."

Input: 'Open the terminal$ and type$ ls -la'
Output: 'Abre la terminal$ y escribe$ ls -la'

Remember: You MUST preserve the number of dollar sign delimiters and their placement in the translated text.
"""
        self.system_prompt = """
You are a professional translator.
Your task is to translate text from English to {target_lang_name}.
The translation MUST be grammatically correct and natural {target_lang_name}.
Only technical terms, commands, URLs, and proper nouns should remain in English.
You must provide a complete translation - returning the original text unchanged is not acceptable.
Return only the sentences translated into {target_lang_name}.
"""

    def get_language_name(self, lang_code: str) -> str:
        """Convert language code to full name."""
        try:
            lang = langcodes.Language.get(lang_code)
            return lang.display_name()
        except Exception as e:
            logger.error(f'Error parsing language code {lang_code}: {e}')
            return lang_code

    def validate_language_code(self, lang_code: str) -> str:
        """Validate and normalize language code."""
        try:
            lang = langcodes.Language.get(lang_code)
            if not lang.is_valid():
                raise ValueError(f'Invalid language code: {lang_code}')
            return lang.language
        except Exception as e:
            logger.error(f'Invalid language code {lang_code}: {e}')
            raise ValueError(f'Invalid language code: {lang_code}')

    def captions_to_transcript(self, entries) -> list:
        """Convert captions to transcript, split by sentences."""
        transcript = ''
        for _, text, _ in entries:
            transcript += text + ' '
        try:
            sentences = nltk.tokenize.sent_tokenize(transcript)
        except LookupError:
            logger.info('Downloading NLTK tokenizer models (one time operation).')
            nltk.download('punkt_tab')
            sentences = nltk.tokenize.sent_tokenize(transcript)
        return sentences

    def parse_vtt(self, file_path: str) -> List[Tuple[str, str, str]]:
        """Parse VTT file into list of (timestamp, text, original_line) tuples."""
        entries = []
        current_time = ''

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == 'WEBVTT' or not line:
                i += 1
                continue

            if '-->' in line:
                current_time = line
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                text = ' '.join(text_lines)
                if text:
                    entries.append((current_time, text, f'{current_time}\n{text}'))
            else:
                i += 1

        # Convert captions to transcript to populate self.sentences list
        self.sentences = self.captions_to_transcript(entries)

        return entries

    def translate_text_chatgpt(self, text: str, target_lang_name: str) -> str:
        """Translate text using ChatGPT API."""

        messages = [
                {'role': 'system', 'content': self.system_prompt.format(target_lang_name=target_lang_name)},
                {'role': 'user', 'content': self.user_prompt.format(text=text,
                                                                    target_lang_name=target_lang_name,
                                                                    exclude_terms=self.exclude_terms)}
                 ]

        try:
            logger.debug(f'Sending request to ChatGPT API for text: {text}...')

            chat_completion = self.client.chat.completions.create(messages=messages, model='gpt-4o')
            response = chat_completion.choices[0].message.content
            return response

        except Exception as e:
            logger.error(f'Translation error: {str(e)}', exc_info=True)
            logger.error(f'Failed text: {text}')
            return text

    def translate_text_anthropic(self, text: str, target_lang_name: str) -> str:
        """Translate text using Anthropic API."""
        tools = [{
            'name': 'translate_vtt',
            'description': self.system_prompt.format(target_lang_name=target_lang_name),
            'input_schema': {
                'type': 'object',
                'properties': {
                    'translated_text': {
                        'type': 'string',
                        'description': f'The text translated into {target_lang_name}'
                    }
                },
                'required': ['translated_text']
            }
        }]

        try:
            logger.debug(f'Sending request to Anthropic API for text: {text}...')

            response = self.client.messages.create(
                model='claude-3-5-sonnet-latest',
                max_tokens=1000,
                temperature=0,
                messages=[{
                    'role': 'user',
                    'content': self.user_prompt.format(text=text,
                                                       target_lang_name=target_lang_name,
                                                       exclude_terms=self.exclude_terms)}
                          ],
                tools=tools
            )

            logger.debug(f'Received response from API: {response}')

            for content in response.content:
                if content.type == 'tool_use':
                    tool_use = content
                    logger.debug(f'Tool use received: {tool_use}')

                    try:
                        # Try to get translation from Anthropic response
                        tool_input = tool_use.input
                        if isinstance(tool_input, dict):
                            translation = tool_input.get('translated_text', '')
                        else:
                            translation = getattr(tool_input, 'translated_text', '')

                        translation = translation.strip()

                        if translation:
                            self.prev_translations.append(translation)
                            if len(self.prev_translations) > 2:
                                self.prev_translations.pop(0)

                            return translation

                    except Exception as e:
                        logger.error(f'Error processing tool response: {e}')
                        logger.debug(f'Tool use structure: {tool_use}')

            logger.warning(f'No valid translation provided for: {text}...')
            return text

        except Exception as e:
            logger.error(f'Translation error: {str(e)}', exc_info=True)
            logger.error(f'Failed text: {text}')
            return text

    def tokenize_vtt(self, entries: List[Tuple[str, str, str]]) -> str:
        """Tokenize VTT entries for translation."""

        # Extract the text from each VTT entry into a list of chunks
        chunks = []
        for entry in entries:
            chunks.append(entry[1])

        # Tokenize chunks into sentences
        sentences = nltk.sent_tokenize(' '.join(chunks))

        # Process each sentence, adding the token $ to the end of each chunk in each sentence
        tokenized_sentences = []
        for sentence in sentences:
            for chunk in chunks:
                if chunk in sentence:
                    sentence = sentence.replace(chunk, chunk + "$")

            tokenized_sentences.append(sentence)

        return tokenized_sentences

    def translate_vtt(self, input_file: str, target_lang: str, output_file: Optional[str] = None) -> None:
        """Translate entire VTT file."""
        # Validate and get full language name
        target_lang = self.validate_language_code(target_lang)
        target_lang_name = self.get_language_name(target_lang)
        logger.info(f'Translating to {target_lang_name} ({target_lang})')

        if not output_file:
            base, ext = os.path.splitext(input_file)
            output_file = f'{base}_{target_lang}{ext}'

        entries = self.parse_vtt(input_file)
        tokenized_text = self.tokenize_vtt(entries)

        # Iterate over tokenized_text, tranlating each sentence
        translated_chunks = []
        for sentence in tqdm.tqdm(tokenized_text, desc='Translating', disable=self.no_progress_bar):
            logger.info(f'Translating sentence: {sentence}')
            translated_sentence = self.translate_text(sentence, target_lang_name)

            # Ensure translated entry count matches tokenized entry count
            if (len(translated_sentence.split('$')) != len(sentence.split('$'))):
                logger.error('Translated entry token count does not match original token count. '
                             f'Original: {len(sentence.split("$"))}, '
                             f'Translated: {len(translated_sentence.split("$"))} - '
                             'Reverting to original English text.')
                logger.error(f'Original: {sentence}')
                logger.error(f'Translated: {translated_sentence}')
                translated_sentence = sentence

            # Clean and detokenize the translated text
            translated_sentence = translated_sentence[0:-1]  # Remove trailing token
            translated_sentence_chunks = translated_sentence.split('$')
            translated_sentence_chunks = [chunk.strip() for chunk in translated_sentence_chunks]

            logger.info(f'Translated sentence: {translated_sentence}')

            # Add translated chunks to list
            translated_chunks.extend(translated_sentence_chunks)

        # Replace the original text with the translated text
        # Check that the number of elements in the translated_chunks list matches the number of entries
        if (len(translated_chunks) != len(entries)):
            raise ValueError('The number of translated chunks does not match the number of entries in the VTT file.')

        for i, (timestamp, _, _) in enumerate(entries):
            translated_chunks[i] = (timestamp, translated_chunks[i])

        # Write the translated VTT file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('WEBVTT\n\n')
            for entry in translated_chunks:
                f.write(f'{entry[0]}\n{entry[1]}\n\n')

        logger.info(f'Translation completed. Output written to: {output_file}')


if __name__ == '__main__':

    if (len(sys.argv) == 1):
        print('vttbabelfish.py: Translate VTT subtitle files using AI\n')
        sys.argv.append('--help')

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input VTT file path')
    parser.add_argument('target_lang', help='Target language (2-letter or 3-letter code, or BCP-47 tag)')
    parser.add_argument('-p', '--platform', help='LLM (anthropic or chatgpt)', default='anthropic')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--api-key', help='Anthropic API key', required=True)
    parser.add_argument('-e', '--exclude-file', help='File with terms not to translate', required=False)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
        no_progress_bar = True
    else:
        logging.getLogger().setLevel(logging.WARNING)
        no_progress_bar = False

    if (args.platform not in ['anthropic', 'chatgpt']):
        logger.error('Only the "anthropic" and "chatgpt" platforms are supported at this time.')
        sys.exit(1)

    if (args.exclude_file):
        logger.info('Excluding terms from file: ' + args.exclude_file)
        with open(args.exclude_file, 'r') as f:
            exclude_terms = f.readlines()
            exclude_terms = ', '.join([x.strip() for x in exclude_terms])
    else:
        exclude_terms = 'Linux, Slingshot, Midnite Meerkats, Falsimentis, command injection, SQL injection'

    translator = VTTTranslator(args.api_key, platform=args.platform,
                               exclude_terms=exclude_terms,
                               no_progress_bar=no_progress_bar)

    try:
        translator.translate_vtt(args.input_file, args.target_lang, args.output)
    except ValueError as e:
        logger.error(str(e))
        print(f'Error: {str(e)}')
        print('\nCommon language codes include:')
        examples = [
            ('en', 'English'),
            ('es', 'Spanish'),
            ('fr', 'French'),
            ('de', 'German'),
            ('it', 'Italian'),
            ('nl', 'Dutch'),
            ('pt', 'Portuguese'),
            ('ru', 'Russian'),
            ('zh', 'Chinese'),
            ('ja', 'Japanese'),
            ('ko', 'Korean')
        ]
        for code, name in examples:
            print(f'{code:4} - {name}')
        print('\nYou can also use BCP-47 tags like "en-US" or "es-ES"')
        sys.exit(1)
