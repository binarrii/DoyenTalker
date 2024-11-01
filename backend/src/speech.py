import os
import requests
import torch
# from TTS.api import TTS
import time
from argparse import ArgumentParser

import edge_tts
import asyncio

import nest_asyncio
nest_asyncio.apply()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)

def generate_speech(path_id, outfile, text, speaker_wav=None, language="en"):
    # Generate the full path for the output file
    output_path = os.path.join(path_id, outfile)
    
    # Generate speech and save to a file
    # tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=language)
    # tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav)

    try:
        _generate_speech_from_edge_tts(output_path, text, language)
    except:
        _generate_speech_from_gptsovits(output_path, text, language)
    
    return output_path

def _generate_speech_from_edge_tts(output_path, text, language="zh-CN-XiaoxiaoNeural"):
    communicate = edge_tts.Communicate(text=text, voice=language, rate="+0%")
    task = communicate.save(output_path)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(task))

def _generate_speech_from_gptsovits(output_path, text, language="zh-cn"):
    url = 'http://192.168.1.73:9880/tts'
    with requests.post(url, json={"text": text}) as resp:
        if resp.status_code == 200:
            with open(output_path, 'wb+') as f:
                for chunk in resp.iter_content(chunk_size=4096):
                    f.write(chunk)
        else:
            raise RuntimeError(f"Error from GPT-SoVITS: {resp.status_code}")

def read_message_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def split_text(text, max_words=100):
    # Split the text into chunks of max_words each
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def speech(args):
    # Create a unique path based on the current timestamp
    path_id = os.path.join("temp", str(int(time.time())))
    os.makedirs(path_id, exist_ok=True)
    
    # Initialize variables
    message = None
    
    # Conditionally read the message file
    if args.message_file:
        message = read_message_from_file(args.message_file)
        
    speaker_wav = args.voice
    language = args.lang
            
        # Generate audio for chunks of text
    audio_files = []
    text_chunks = split_text(message)  # Function to split text into chunks
    

    for i, message in enumerate(text_chunks):
        outfile = f"output_part_{i + 1}.wav"
        try:
            output_path = generate_speech(path_id, outfile, text=message, speaker_wav=speaker_wav, language=language)
            audio_files.append(output_path)
        except Exception as e:
            print(f"An error occurred while generating audio for text {i + 1}: {e}")


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--message_file", type=str, help="path to the file containing the speech message")
    parser.add_argument("--voice", type=str, default='./assets/voice/ab_voice.mp3', help="path to speaker voice file")
    parser.add_argument("--lang", type=str, default='en', help="select the language for speaker voice option are (en - English , es - Spanish , fr - French , de - German , it - Italian , pt - Portuguese , pl - Polish , tr - Turkish , ru - Russian , nl - Dutch , cs - Czech , ar - Araic , zh-cn - Chinese (Simplified) , hu - Hungarian , ko - Korean , ja - Japanese , hi - Hindi)")
   
    args = parser.parse_args()

    speech(args)
