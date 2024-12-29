import sys
import glob
import os
import random
import json
import argparse
import pydub

def main(args):

    # path = "/content/drive/MyDrive/STT main/LibriSpeech/dev-clean"
    path = "LibriSpeech/dev-clean"
    folders = os.listdir(path)

    maxi = float(0)
    data = []

    print("Traversing all the available files in LibriSpeech...")

    for folder in folders:
        print(f"folder = {folder}")
        subfolders = os.listdir(path + '/' +folder)
        # print(subfolders)
        for subfolder in subfolders:
            # print(subfolder)
            flacFiles = glob.glob(path + '/' + folder + '/'+ subfolder + "/*.flac")
            transctipt = glob.glob(path + '/' + folder + '/'+ subfolder + "/*.txt")
            with open(transctipt[0] , 'r') as script:
                content = script.read().lower()
            
            # print(content)
            flacFiles.sort()
            for i,flac in enumerate(flacFiles):
                file_name = flac.split('/')[4].split('.')[0]
                # print(f"flac = {flac}" )
                # print(f"file_name = {file_name}")
                # data, samplerate = sf.read(flac)
                # maxi = max(maxi,data.shape[0]/samplerate)
                text = content.split('\n')[i].split(file_name)[1]
                text = text.strip()

                sound = pydub.AudioSegment.from_file(path + '/' + folder + '/'+ subfolder + '/' + file_name + ".flac")
                sound.export("dest_wav_file"+ '/' + file_name + ".wav",format = "wav")

                data.append(
                    { "key" : "dest_wav_file" + '/' + file_name + ".wav",
                        "text" : text
                    }
                )

    # print(maxi)

    random.shuffle(data) 
    print("creating Json file !!!")

    save_path = args.save_json_location
    train_percentage = args.train_test_ratio
    train_len = int(len(data)*train_percentage)
    test_len = int(len(data) - train_len)

    i = int(0)
    with open(save_path + "/" + "train.json", "w") as f:
        f.write("[\n")
        while(i < train_len):
            r = data[i]
            line = json.dumps(r)
            if i == train_len -1:
                f.write(line + "\n")
            else: f.write(line + ",\n")
            i = i + 1
        f.write("]")

    with open(save_path + "/" + "test.json", "w") as f:
        j = int(0)
        f.write("[\n")
        while(j < test_len and i < len(data)):
            r = data[i]
            line = json.dumps(r)
            if(j == test_len -1):
                f.write(line + "\n")
            else: f.write(line + ",\n")
            i = i +1
            j = j +1
        f.write("]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_json_location', default= None , required= True, help= "Path where you want to save train and test json files")
    parser.add_argument('--train_test_ratio', default= 0.9,required=False,help= "Ratio of train to test (0-1)")

    args = parser.parse_args()

    main(args)