""" ========================================================================================================
create fastapi app 
========================================================================================================="""
from fastapi import FastAPI
app = FastAPI()


""" ========================================================================================================
define structure for requests (Pydantic & more)
========================================================================================================="""
from fastapi import Request # for get
from pydantic import BaseModel # for post

class FinalFrameInSeqRequest(BaseModel):
	video_name: str
    frame_num: int

class VideoRequest(BaseModel):
	video_name: str
    video_data: int
""" ========================================================================================================
custom modules
========================================================================================================="""



""" ========================================================================================================
routes
========================================================================================================="""
@app.post("/predict")
def predict_from_final_frame_in_seq(frame_req: FinalFrameInSeqRequest):
    pass
	
@app.post("/upload")
def upload_video(req: VideoRequest):
    pass
