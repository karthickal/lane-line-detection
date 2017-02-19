# import required libraries
from moviepy.editor import VideoFileClip
from detector import LaneDetector
import utils

if __name__ == "__main__":

    # create a LaneDetector object and load the video
    lane_detector = LaneDetector()
    output = 'project_video_test.mp4'
    source = VideoFileClip('project_video.mp4')

    # find the lane lines for each of the frame
    clip = source.fl_image(lane_detector.run)
    clip.write_videofile(output, audio=False)

    print("{} frames were processed.".format(lane_detector.processed_frames))
