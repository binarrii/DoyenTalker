import re
import shutil
import subprocess
import torch
import time 
import os, sys, time
from argparse import ArgumentParser
import datetime as dt
import humanize
from moviepy.editor import concatenate_videoclips, VideoFileClip

from src.speech import generate_speech
from src.utils.preprocess import CropAndExtract
from src.audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_audio_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils import audio


def split_text(text, max_words=100):
    # Split the text into chunks of max_words each
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def main(args):
    tstart = time.time()
    
    # Function to read the content of the message file
    def read_message_from_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    input_voice = args.voice
    input_lang = args.lang
    
    path_id = args.task_id or str(int(time.time()))
    path = os.path.join(args.result_dir,path_id)
    
    print("path_id:", path_id, "path:", path)
    os.makedirs(path, exist_ok=True)
    
    # Generate audio for chunks of text
    audio_files = []
    tspeech = 0

    # Conditionally read the message file
    if args.message_file:
        message = read_message_from_file(args.message_file)
        text_chunks = split_text(message)
        tspeech_start = time.time()  # Start timing speech generation
        print("-----------------------------------------")
        print("generating speech")
        for i, chunk in enumerate(text_chunks):
            audio_part = f"output_part_{i + 1}.t.wav"
            audio_path = os.path.join(path, audio_part)
            try:
                generate_speech(path, audio_part, chunk, input_voice, input_lang)
                audio_files.append(audio_path)
                print(f">>>>> audio_files: {len(audio_files)}")
                if args.audio_file:
                    total_duration = 0
                    for a in audio_files:
                        total_duration += audio.get_duration(a)
                    print(f">>>>> total_duration: {total_duration}")
                    target_duration = audio.get_duration(args.audio_file)
                    print(f">>>>> target_duration: {target_duration}")
                    stretch_ratio = total_duration / target_duration
                    print(f">>>>> stretch_ratio: {stretch_ratio}")
                    audio_files_new = []
                    for a in audio_files:
                        ao = a.replace('.t.wav', '.mp3')
                        audio.change_duration(a, ao, stretch_ratio=stretch_ratio)
                        audio_files_new.append(ao)
                    audio_files = audio_files_new
                    print(f">>>>> audio_files_new: {len(audio_files)}")
            except Exception as e:
                print(f"An error occurred while generating audio for text {i + 1}: {e}")
        tspeech_end = time.time()
        tspeech = tspeech_end - tspeech_start
    else:
        audio_files.append(args.audio_file)
    
    video_files = []

    pic_path = args.avatar_image
    save_dir = path
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    doyentalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    # Init models
    preprocess_model = CropAndExtract(doyentalker_paths, device)
    audio_to_coeff = Audio2Coeff(doyentalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(doyentalker_paths, device)

    # Crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    timage_start = time.time()
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, avatar_image_flag=True, pic_size=args.size)
    
    timage_end = time.time()
    timage = timage_end - timage_start
    
    ## only in use if the reference video in given 
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, avatar_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, avatar_image_flag=False)
    else:
        ref_pose_coeff_path = None
        
    # coeff2video
    tanimate_start = time.time()
    
    # Process each audio file
    for i, audio_path in enumerate(audio_files):
        # Generate video for the current audio file
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        
        # 3dface render
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, f'3dface_part_{i + 1}.mp4'))
        
        
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                   batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                   expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

        result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                             enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
        tanimate_end = time.time()
        tanimate = tanimate_end - tanimate_start
        
        # Save video path
        video_file = os.path.join(save_dir, f'generated_video_part_{i + 1}.mp4')
        shutil.move(result, video_file)
        video_files.append(video_file)
        print(f'Video for part {i + 1} is named:', video_file)
        
    
    # Combine all video files
    tcombine_video_start = time.time()  
    combined_video_path = os.path.join(save_dir, path_id + ".mp4")
    clips = [VideoFileClip(v) for v in video_files]
    
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(combined_video_path, codec="libx264")
    print(f">>>>> final_clip: {final_clip.duration}")
    
    tcombine_video_end = time.time()  
    t_combine_video = tcombine_video_end - tcombine_video_start
    
    combined_video_path = shutil.move(combined_video_path, args.result_dir)
    print('The generated video is named:', combined_video_path)

    if args.audio_file and args.message_file:
        final_output_video_path = f"{args.workdir}/{os.path.basename(combined_video_path)}"
        subprocess.call([
            'ffmpeg',
            '-i', os.path.abspath(combined_video_path),
            '-i', os.path.abspath(args.audio_file),
            '-c:v', 'copy',
            '-map', '0:v',
            '-map', '1:a',
            '-shortest',
            '-y',
            final_output_video_path
        ])
        os.remove(combined_video_path)
        shutil.move(final_output_video_path, combined_video_path)


    if not args.verbose:
        shutil.rmtree(save_dir)


    print("done")
    print("Overall timing")
    print("--------------")
    print("generating speech:", humanize.naturaldelta(dt.timedelta(seconds=tspeech)))
    print("generating avatar image:", humanize.naturaldelta(dt.timedelta(seconds=timage)))
    print("animating face:", humanize.naturaldelta(dt.timedelta(seconds=tanimate)))
    print("Combined video:", humanize.naturaldelta(dt.timedelta(seconds=t_combine_video)))
    print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--task_id", type=str, default=None, help="the id of this task")
    parser.add_argument("--message_file", type=str, default=None, help="path to the file containing the speech message")
    parser.add_argument("--audio_file", type=str, default=None, help="path to driving audio")
    parser.add_argument("--voice", type=str, default='./assets/voice/ab_voice.mp3', help="path to speaker voice file")
    parser.add_argument("--lang", type=str, default='en', help="select the language for speaker voice option are (en - English , es - Spanish , fr - French , de - German , it - Italian , pt - Portuguese , pl - Polish , tr - Turkish , ru - Russian , nl - Dutch , cs - Czech , ar - Araic , zh-cn - Chinese (Simplified) , hu - Hungarian , ko - Korean , ja - Japanese , hi - Hindi)")
    parser.add_argument("--avatar_image", default='./assets/avatar/male1.jpeg', help="path to avatar image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.5, help="a larger value will make the expression motion stronger (max 3.0).")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer] to enhance the generated face via face restoration network")
    parser.add_argument('--background_enhancer', type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", type=bool, default=True, help="using the same pose parameters as the original image, fewer head motion.") 
    parser.add_argument("--preprocess", default='full', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the image") 
    parser.add_argument("--verbose", action="store_true", help="saving the intermediate output or not") 
    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version") 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc', default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)
