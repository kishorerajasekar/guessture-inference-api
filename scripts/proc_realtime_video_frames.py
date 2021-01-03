import config, os, cv2, tqdm, json
from utils.file.name import get_extension_from
from utils.file.manager import create_folder
from utils.mediapipe.webcam import pose, hands
from utils.mediapipe.defaults import mp_pose, mp_hands, mp_drawing
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
        self.xtrain = []


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
    # add process functions for each individual frames here ....
    def proc_posepoints_json(self, frame, frame_num, is_final_frame):
        results = pose.process(frame)
        results_h = hands.process(frame)

        points = []
        for data_point in results.pose_landmarks.landmark[11:24]:
            points.append(data_point.x)
            points.append(data_point.y)
            print("body found ..")
        if results_h.multi_hand_landmarks:
            for key,hand_landmark in enumerate(results_h.multi_hand_landmarks):
                for data_point in results_h.multi_hand_landmarks[key].landmark:
                    points.append(data_point.x)
                    points.append(data_point.y)
            print("hands found ..")
        
        self.__results_h = results_h if results_h.multi_hand_landmarks else None
        self.__results = results
        self.xtrain.append(points)
        
        if is_final_frame:
            print("[DEBUG] Finally saving ... ")
            create_folder(self.output_dir+"posepoints_json")
            with open(self.output_dir+'posepoints_json/data.json', 'w') as fp:
                json.dump(self.xtrain, fp)


    def proc_save_frames(self, frame, frame_num, is_final_frame):
        if frame_num == 0: create_folder(self.output_dir+"frames")
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, self.__results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if self.__results_h:
            for hand_landmarks in self.__results_h.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(self.output_dir + "frames/" + f"{frame_num}".zfill(10) + ".png", annotated_image)


    def proc_db_records(self, frame, frame_num, is_final_frame):
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
            self.proc_posepoints_json(frame, frame_num ,self.__is_final_frame(frame_num))
            self.proc_save_frames(frame, frame_num, self.__is_final_frame(frame_num))


    def __is_final_frame(self, cur_frame_num):
        return cur_frame_num == self.n_frames-2
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
    process_all_video_frames()