from flask import Flask, jsonify, request, make_response
import json
import boto3
from predict_ver_2 import Model_test
import time
import datetime

# 20210122  Image Resizing
from PIL import Image


# Flask
app = Flask(__name__)


@app.route('/picture', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 0. 현재 년월일_시간
        # now = datetime.datetime.now().strftime("%y%m%d_%H%M")

        # 1. request로부터 사진 URL을 받아온다.
        url = request.get_json()['origin']
        filename = request.get_json()['filename']

        try:
            # 2. boto3을 이용해서 사진을 받아서, dataset/new_images에 업로드
            bucketname = "poohstorybucket"
            s3 = boto3.resource('s3', aws_access_key_id="key",
                                aws_secret_access_key="key")
            s3.Bucket(bucketname).download_file(filename, "dataset/new_images/temp/" + filename)
            img = Image.open('dataset/new_images/temp/' + filename)
            img_resize = img.resize((900,1200))
            img_resize.save('dataset/new_images/' + filename + '.jpg')

        except FileNotFoundError:
            return jsonify({"errorcode": -1})
        except Exception:
            return jsonify({"errorcode": -1})

        try:
            # 3. 사진을 불러온다.
            # 4. 불러온 사진을 딥러닝 모델에 넣는다.
            Model_test(filename)

            list = ["Adaboost_correct_for_outlier", "Adaboost_poop_color_with", "Adaboost_poop_color_without",
                    "Adaboost_segmentation", "Bayesian_correct_for_outlier", "Bayesian_poop_color_with",
                    "Bayesian_poop_color_without", "Bayesian_segmentation", "Linear_correct_for_outlier",
                    "Linear_poop_color_with", "Linear_poop_color_without", "Linear_segmentation"]

            for i in list:
                data = open("result/" + i + "/" + filename + ".jpg", 'rb')
                image_name = i + "_" + filename
                if i == "Bayesian_correct_for_outlier":
                    imageurl = "/" + image_name
                s3.Bucket(bucketname).put_object(Body=data, Key=image_name, ACL='public-read',
                                                 ContentType='image/jpeg')

            # 8. 주소를 response로 보낸 후, status 200 세팅한다.
            imageurl = "/" + "Bayesian_correct_for_outlier_" + filename

        except IndexError:
            return jsonify({"errorcode": -1})
        except ValueError:
            return jsonify({"errorcode": -1})
        except FileNotFoundError:
            return jsonify({"errorcode": -1})
        except Exception:
            return jsonify({"errorcode": -1})

        return jsonify({'image': imageurl})

    else:
        return jsonify({"errorcode": -1})


# Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=False)
