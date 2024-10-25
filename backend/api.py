import os
import signal
import subprocess
import time
import traceback
import uuid
import cv2
import torch
import uvicorn

import numpy as np

from argparse import Namespace
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from main import main as execute_doyentalker

tmp_path = "/tmp/doyen"
os.makedirs(tmp_path, exist_ok=True)

app = FastAPI()


@app.post("/api/synthesis")
async def assets(request: Request):
    content_type = request.headers.get('Content-Type')
    if content_type.startswith('multipart/form-data'):
        req_form = await request.form()
    else:
        raise HTTPException(status_code=400, detail='Requires form data')
    
    text = req_form.get("text", None)
    if text:
        message_file = f"{tmp_path}/{uuid.uuid4().hex}.txt"
        with open(message_file, 'w') as f:
            f.write(text)
    else:
        message_file = None

    portrait = req_form.get("portrait", None)
    if portrait:
        portrait_image_path = f"{tmp_path}/{portrait.filename}"
        with open(portrait_image_path, 'wb') as f:
            raw_bytes = await portrait.read()
            f.write(raw_bytes)
            
            # img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            # h, w = img_bgr.shape[:2]
            # dlimit = 1024
            # if w > dlimit:
            #     img_bgr = cv2.resize(img_bgr, (dlimit, int(h / (w / dlimit))))
            # elif h > dlimit:
            #     img_bgr = cv2.resize(img_bgr, (int(w / (h / dlimit)), dlimit))
            # _, buffer = cv2.imencode(".jpeg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # f.write(buffer.tobytes())
    else:
        raise HTTPException(status_code=400, detail='Portrait image is required')

    audio = req_form.get("audio", None)
    if audio:
        audio_path = f"{tmp_path}/{audio.filename}"
        with open(audio_path, 'wb+') as f:
            while True:
                chunk = await audio.read(1 << 13)
                if not chunk:
                    break
                f.write(chunk)
    else:
        audio_path = None

    if not audio and not text:
        raise HTTPException(status_code=400, detail='Audio or text is required')

    task_id = str(int(time.time()))
    lang = req_form.get("lang", None) or "zh-CN-XiaoxiaoNeural"
    voice = req_form.get("voice", None) or "assets/aijia_ref.mp3"

    args = Namespace(
        task_id=task_id,
        message_file=message_file,
        audio_file=audio_path,
        voice=voice,
        lang=lang,
        avatar_image=portrait_image_path,
        ref_eyeblink=None,
        ref_pose=None,
        checkpoint_dir="./checkpoints",
        result_dir="./results",
        pose_style=0,
        batch_size=2,
        size=512,
        expression_scale=1.5,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        enhancer=None,
        background_enhancer=None,
        cpu=False,
        face3dvis=False,
        still=True,
        preprocess="extcrop",
        verbose=False,
        old_version=False,
        net_recon="resnet50",
        init_path=None,
        use_last_fc=False,
        bfm_folder="./checkpoints/BFM_Fitting/",
        bfm_model="BFM_model_front.mat",
        focal=1015.,
        center=112.,
        camera_d=10.,
        z_near=5.,
        z_far=15.,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        output_dir = os.path.join('results')
        os.makedirs(output_dir, exist_ok=True)
        
        # subprocess.run(command, check=True, cwd=os.path.dirname(__file__))
        execute_doyentalker(args=args)
        
        output_video_path = os.path.join(output_dir, f"{task_id}.mp4")
        print(f">>>>>>>>>>>>>>>>>>>> {output_video_path}")
    except subprocess.CalledProcessError as e:
        output_video_path = ""
        print(f"Error running DoyenTalker: {e}")

    def iterfile():
        with open(output_video_path, mode='rb') as f:
            yield from f

    return StreamingResponse(iterfile(), media_type='video/mp4')


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9100, workers=1)
    except Exception as e:
        traceback.print_stack()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
