""" ========================================================================================================
create fastapi app 
========================================================================================================="""
from fastapi import FastAPI, File ,UploadFile
app = FastAPI()


""" ========================================================================================================
CORS
========================================================================================================="""
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "https://rakesh4real.github.io",
    "http://rakesh4real.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


""" ========================================================================================================
define structure for requests (Pydantic & more)
========================================================================================================="""
from fastapi import Request # for get
from pydantic import BaseModel # for post

class FinalFrameInSeqRequest(BaseModel):
    file_name: str = "Not optional!"
    frame_num: int = -1
""" ========================================================================================================
custom modules
========================================================================================================="""


""" ========================================================================================================
custom modules
========================================================================================================="""
import config
from utils.file.name import get_extension_from
from scripts.proc_realtime_video_frames import process_single_video_frames    
from scripts.infer_from_posepoints_json import ProcessPosePointsJSON

json_processors = {}
def set_json_processor(video_filename):
    json_processors[video_filename] = ProcessPosePointsJSON(
        config.REAL_TIME_VIDEOS_OUTPUT_DIR + video_filename)

def get_json_processor(video_filename):
    return json_processors[video_filename]

""" ========================================================================================================
routes
========================================================================================================="""
@app.post("/uploadfile/")
def create_upload_file(file: UploadFile = File(...)):
    """
    - uploads file in inputs directory
    - Generates frames
    - Generates json data of pose points
    - Initializes json_processor for realtime infrerence
    """
    if get_extension_from(file.filename) in config.ALLOWED_VIDEO_EXTENSTIONS:
        video_in_bytes = file.file.read()

        # i. save in input dir
        with open(config.REAL_TIME_VIDEOS + file.filename, 'wb') as fp:
            fp.write(video_in_bytes)

        # ii. genereate json, frames etc ..
        process_single_video_frames(file.filename)

        # iii. initialize json processor for prediction
        set_json_processor(file.filename)

        status = "Sucess" if video_in_bytes is not None else "Fail"
        return {"status": status}
    return {"status": "File Extension Not Allowed :("}


@app.post("/predict")
def predict_from_final_frame_in_seq(frame_req: FinalFrameInSeqRequest):
    """
    takes frame_num and video name as input and predicts class for 
    the associated frame number
    """

    dist, pred = get_json_processor(frame_req.file_name)\
                    .predict(frame_req.frame_num)
    return {
        "dist" : dist[0][0], 
        "pred" : pred[0]
    }
