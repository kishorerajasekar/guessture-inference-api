import config, os, cv2, tqdm
from utils.file.name import get_extension_from
from utils.file.manager import create_folder
from tqdm import tqdm



class VideoFramesProcessor:
    # ==========================================================================================================
    # beg: constructor and frames generator
    # ==========================================================================================================    
    def __init__(self, filename):
        
        self.filename = filename
        self.ext = get_extension_from(filename)
        self.path = config.REAL_TIME_VIDEOS + filename
        
        self.output_dir = config.REAL_TIME_VIDEOS_OUTPUT_DIR + self.filename + "/"
        create_folder(self.output_dir)

        self.cap = cv2.VideoCapture(self.path)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)


    def get_next_frame(self):
        frame_num = -1
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            frame_num += 1
            yield frame_num, frame
    # ==========================================================================================================
    # end: constructor and frames generator
    # ==========================================================================================================    


    # ==========================================================================================================
    # beg: processes for frames    
    # ==========================================================================================================    
    def proc_posepoints_csv(self, frame, frame_num):
        create_folder(self.output_dir+"posepoints_csv")


    def proc_save_frames(self, frame, frame_num):
        pass


    def proc_db_records(self, frame, frame_num):
        pass
    # ==========================================================================================================
    # end: processes for frames    
    # ==========================================================================================================    


    # ==========================================================================================================
    # beg: controller for processes    
    # ==========================================================================================================    
    def start(self):
        """
        define the processes to run
        """

        for frame_num, frame in tqdm(self.get_next_frame(), total=self.n_frames, desc=self.filename):
            
            # add / remove processes from here
            self.proc_posepoints_csv(frame, frame_num)
            self.proc_save_frames(frame, frame_num)
    # ==========================================================================================================
    # beg: controller for processes    
    # ==========================================================================================================    



def process_all_video_frames():
    for filename in os.listdir(config.REAL_TIME_VIDEOS):
        if get_extension_from(filename) in config.ALLOWED_VIDEO_EXTENSTIONS:

            processor = VideoFramesProcessor(filename)
            processor.start()


def process_single_video_frames(filename):
    if get_extension_from(filename) in config.ALLOWED_VIDEO_EXTENSTIONS:
        processor = VideoFramesProcessor(filename)
        processor.start()


if __name__ == '__main__':
    process_video_frames()