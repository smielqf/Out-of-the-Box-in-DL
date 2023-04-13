### This script is used for subtitling videos automatically.
### The skeleton is designed as following:
### 1. Spilt audio and frames
### 2. Convert audio to text with time stamps
### 3. Merge text into videos
### Language Support: Chinese

import os
import math
from time import strftime, gmtime
from tqdm import tqdm

from modelscope.pipelines import pipeline as m_pipeline
from modelscope.utils.constant import Tasks
import librosa

import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from pydub import AudioSegment

from typing import List


def generate_srt_online(words:List[str], time_stamps:List[List[float,float]], output:str, max_sent_len:int = 20):
    """Given lists of words and corresponding timestamps, it can generate a srt file.

    Parameters
    ----------
    words : List[str]
        A list containing words in Chinese.
    time_stamps : List[List[float,float]]
        A list containing timestamps. The first value represents the beginning time of pronouced word while 
        the second one is the ending time.
    output : str,
        The filename of target srt file, which includes the path optionally.
    max_sent_len : int, optional
        A number controls the length of sentence occurring in the video, by default 20 for Chinese.
    """    
    def format_time(timestamp):
        if timestamp is None:
            return ''
        second = math.modf(timestamp / 1000)
        return strftime('%H:%M:%S', gmtime(second[1])) + ',' + str(int(round(second[0], 3) * 1000))
   
    # If the gap between two words is larger than 400, it is recognized as an endding of one sentence 
    # and a beginning of another sentence.
    interval_length = 400
    sentence = ''
    front = 0
    sentence_list = []
    sentence_timestamps = []

    for w, (b, e) in zip(words, time_stamps):
        if sentence == '':
            sentence += w
            sentence_timestamps.append([format_time(b), format_time(e)])
        elif len(sentence) >= max_sent_len or b - front > interval_length:
            sentence_list.append(sentence)
            sentence = ''
            sentence += w
            sentence_timestamps.append([format_time(b), format_time(e)])
        else:
            sentence += w
            sentence_timestamps[-1][1] = format_time(e)
        
        if len(sentence) >= max_sent_len:
            sentence_list.append(sentence)
            sentence = ''
            sentence += w
            sentence_timestamps.append([format_time(b), format_time(e)])
        front = e

    with open(output, 'w', encoding='utf-8') as f:
        count = 1
        for sentence, (t_b, t_e) in zip(sentence_list, sentence_timestamps):
            f.write(str(count) + '\n')
            f.write(t_b + ' -->' + t_e + '\n')
            f.write(sentence + '\n')
            f.write('\n')
            count += 1

def split_audio_from_video(video_name, interval_len=60):
    """Given a video, it is necessary to clip its audio to multiple smaller clipped audios, since too long
    audio can lead to overflow of memory or video memory.

    Parameters
    ----------
    video_name : str,
        The filename of the given video.
    interval_len : int, optional
        A number controlling the length of clipped audio, by default 60. The smaller it is, the less memory or video
        memory is required.

    Returns
    -------
    [str,],
        A list containing filenames of clipped audios.
    """    
    prefix_name = ''.join(video_name.split('.')[:-1])
    video = mp.VideoFileClip(video_name)
    tmp_audio_name = prefix_name + "_tmp.wav"
    video.audio.write_audiofile(tmp_audio_name)
    
    unit_in_sencond = 1000
    tmp_audio = AudioSegment.from_wav(tmp_audio_name)
    time_stamp = 0
    audio_slice_list = []
    while time_stamp < video.duration:
        if time_stamp + interval_len > video.duration:
            audio_slice = tmp_audio[time_stamp*unit_in_sencond: video.duration*unit_in_sencond]
        else:    
            audio_slice = tmp_audio[time_stamp*unit_in_sencond: (time_stamp+interval_len)*unit_in_sencond]
        audio_slice_name = prefix_name + '_' + str(time_stamp) + '.wav'
        audio_slice_list.append(audio_slice_name)
        audio_slice.export(audio_slice_name, format='wav')
        time_stamp += interval_len
    
    os.remove(tmp_audio_name)
    return audio_slice_list

def convert_audio_to_text(audio_list:List[str], language:str, interval_len:int, output:str, max_sent_len:int, device:str='cpu'):
    """Given a list of audio, it can perform auto speech recogintion and generate a srt file.

    Parameters
    ----------
    audio_list : List[str]
        A list containing multiple audios. Usually, each audio should not be too long.
    language : str
        The language to recognize.
    interval_len : int
        A number determining the length of clipped audio. It is used to calculate timestamps of words.
    output : str
        The filename of the srt file.
    max_sent_len : int
        A number determining the length of sentence occurring in the video.
    device : str, optional
        The hardware for running ASR models, by default 'cpu'. If GPU is available, it can be set as "gpu".

    Returns
    -------
    str
        The filename of the generated srt file.
    """        

    if language == 'english':
        pass
    elif language == 'chinese':
        total_text = []
        total_timestamps = []
        for i in tqdm(range(len(audio_list))):
            audio_input, _ = librosa.load(audio_list[i], sr=16_000)
            """"
                The ASR model for Chinses is  "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch". 
                Developers can replace it with newest open source model in ModelScope or Hugging Face.
                "damo/speech_timestamp_prediction-v1-16k-offline" is used to obtain the time stamp of pronounced words for
                synthetizing the subtitle.
            """
            p = m_pipeline(task=Tasks.auto_speech_recognition, 
                            model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 
                            timestamp_model="damo/speech_timestamp_prediction-v1-16k-offline", 
                            timestamp_model_revision="v1.0.5", 
                            device=device,)
            res = p(audio_input)
            if 'text' in res:
                total_text += res['text'].split(' ')
                total_timestamps += [[x + interval_len * 1000 * i, y + interval_len * 1000 * i] for x, y in res['timestamp']]

        generate_srt_online(total_text, total_timestamps, output=output, max_sent_len=max_sent_len)
    elif language == 'cross':
        pass

    return output

def merge_text_into_video(video_name:str, subtitle:str, output:str, font:str, fontsize:int, color:str):
    """
        The code is following the example given in https://zulko.github.io/moviepy/_modules/moviepy/video/tools/subtitles.html.

    Parameters
    ----------
    video_name : str
        The filename of the given video.
    subtitle : str
        The filename of the srt file of given video.
    output : str
        The filename of the target video.
    font : str
        The font of the subtitle.
    fontsize : int
        The size of words occurring in the video.
    color : str
        The color of words occurring in the video.
    """ 
    generator = lambda txt: mp.TextClip(txt, font=font, fontsize=fontsize, color=color)    
    subtitles = SubtitlesClip(subtitle, generator)
    video = mp.VideoFileClip(video_name)
    result = mp.CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
    result.write_videofile(output, fps=video.fps)



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--color", type=str, default='black', choices=['black', 'white', 'red', 'blue'], help='The color of subtitles. Default color is set as black.')
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'gpu'], help='The device for inferring the audio of the source video. Default device is "cpu". If GPU is available, "gpu" is ok.')
    parser.add_argument('--font', type=str, default='', help='A string representing the font or the path of stored font.')
    parser.add_argument('--font-size', type=int, default=60, help='The font size of subtitles. Default font size is set as 60.' )
    parser.add_argument('--interval_len', type=int, default=30, help='A parameter to control the length of temporally clipped video, e.g., 30s. The lower it is, the less video memory is required.')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english', 'cross'], help='The language of subtitle. Default language is "chinese".')
    parser.add_argument('--max-sent-len', type=int, default=20, help='The maximum length of a sentence in the subtitle. Default length is set as 20 for "chinese".')
    parser.add_argument('--output', type=str, default='', help='The name of target video after subtitling without extention name, e.g., "test_output". If it was not assigned, "_subtitle" is concatenated following the name of source video.')
    parser.add_argument('--video', type=str, required=True, help='The source video for subtitling, e.g., "test.mp4')

    args = parser.parse_args()

    if args.language == 'chinese' and args.font == '':
        args.font = './font/simkai.ttf'
    elif args.language == 'english' and args.font == '':
        args.font = 'Arial'
    
    return args

if __name__ == '__main__':

    args = parse_args()

    audio_slice_list = split_audio_from_video(video_name=args.video, interval_len=args.interval_len)
    if args.output == '':
        output_subtitle = ''.join(args.video.split('.')[:-1]) + '_subtitle.txt'
    else:
        output_subtitle = args.output + '.txt'
    
    convert_audio_to_text(audio_slice_list, language=args.language, interval_len=args.interval_len, output=output_subtitle, max_sent_len=args.max_sent_len, deice=args.device)
    for audio_slice in audio_slice_list:
        os.remove(audio_slice)
    
    if args.output == '':
        output = ''.join(args.video.split('.')[:-1]) + '_subtitle.' + args.video.split('.')[-1]
    else:
        output = args.output + '.' + args.video.spilt('.')[-1]
    merge_text_into_video(video_name=args.video, subtitle=output_subtitle, output=output, font=args.font, fontsize=args.font_size, color=args.color)
    
