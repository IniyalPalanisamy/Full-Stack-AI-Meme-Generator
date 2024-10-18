#!/usr/bin/env python3
# AI Meme Generator
# Creates memes using various AI service APIs. OpenAI's chatGPT to generate the meme text and image prompt, 
# and several optional image generators for the meme picture. Then combines the meme text and image into a meme using Pillow.
# Author: ThioJoe
# Project Page: https://github.com/ThioJoe/Full-Stack-AI-Meme-Generator

version = "1.0.5"

# Import installed libraries
import openai
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image, ImageDraw, ImageFont
import requests

# Import standard libraries
import warnings
import re
from base64 import b64decode
from pkg_resources import parse_version
from collections import namedtuple
import io
from datetime import datetime
import glob
import os
import textwrap
import sys
import argparse
import configparser
import platform
import shutil
import traceback

# =============================================== Argument Parser ================================================
# Parse the arguments at the start of the script
parser = argparse.ArgumentParser()
parser.add_argument("--openaikey", help="OpenAI API key")
parser.add_argument("--clipdropkey", help="ClipDrop API key")
parser.add_argument("--stabilitykey", help="Stability AI API key")
parser.add_argument("--userprompt", help="A meme subject or concept to send to the chat bot.")
parser.add_argument("--memecount", type=int, default=1, help="The number of memes to create.")
parser.add_argument("--imageplatform", default='clipdrop', help="The image platform to use. Options: 'openai', 'stability', 'clipdrop'")
parser.add_argument("--temperature", type=float, default=1.0, help="The temperature to use for the chat bot.")
parser.add_argument("--basicinstructions", help="Basic instructions for the chat bot.")
parser.add_argument("--imagespecialinstructions", help="Special instructions for the image prompt.")
parser.add_argument("--nouserinput", action='store_true', help="Prevent any user input prompts.")
parser.add_argument("--nofilesave", action='store_true', help="If specified, the meme will not be saved to a file.")
args = parser.parse_args()

# Create a namedtuple for API keys
ApiKeysTupleClass = namedtuple('ApiKeysTupleClass', ['openai_key', 'clipdrop_key', 'stability_key'])

# Custom exceptions
class NoFontFileError(Exception):
    def __init__(self, message, font_file):
        full_error_message = f'Font file "{font_file}" not found. Please add the font file to the same folder as this script.'
        super().__init__(full_error_message)
        self.font_file = font_file
        self.simple_message = message

class MissingOpenAIKeyError(Exception):
    def __init__(self, message):
        full_error_message = "No OpenAI API key found. Please add your OpenAI API key to the api_keys.ini file."
        super().__init__(full_error_message)
        self.simple_message = message    

class MissingAPIKeyError(Exception):
    def __init__(self, message, api_platform):
        full_error_message = f"{api_platform} API key not found in the api_keys.ini file."
        super().__init__(full_error_message)
        self.api_platform = api_platform
        self.simple_message = message

class InvalidImagePlatformError(Exception):
    def __init__(self, message, given_platform, valid_platforms):
        full_error_message = f"Invalid image platform '{given_platform}'. Valid platforms are: {valid_platforms}"
        super().__init__(full_error_message)
        self.given_platform = given_platform
        self.valid_platforms = valid_platforms
        self.simple_message = message

# ==============================================================================================

def construct_system_prompt(basic_instructions, image_special_instructions):
    """Constructs the system prompt for the chatbot."""
    format_instructions = (
        "You are a meme generator with the following formatting instructions. "
        "Each meme will consist of text that will appear at the top, and an image to go along with it. "
        "The user will send you a message with a general theme or concept on which you will base the meme. "
        "You must respond only in the format as described next, because your response will be parsed. "
        "The first line of your response should be: 'Meme Text: ' followed by the meme text. "
        "The second line of your response should be: 'Image Prompt: ' followed by the image prompt text."
    )
    system_prompt = (
        f"{format_instructions} Next are
