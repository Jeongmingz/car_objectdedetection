import boto3, json, cv2

with open('access.json', 'r') as file:
    key = json.load(file)


client = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION
                      )


file_name = 'result/img/result.jpg'      # 다운될 이미지 이름
bucket = 'car-object-detaction'           #버켓 주소
key = 'test.jpg' # s3 이미지

client.download_file(bucket, key, file_name)
