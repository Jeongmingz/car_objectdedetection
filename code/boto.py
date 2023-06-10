import boto3

AWS_ACCESS_KEY_ID ="AKIA52YNHYEV7ALFHPF6"
AWS_SECRET_ACCESS_KEY = "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"
AWS_DEFAULT_REGION = "ap-northeast-2"
client = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION
                      )

file_name = 'base.png'      # 업로드할 파일 이름
bucket = 'car-object-detaction'           #버켓 주소
key = 'test.jpg' # s3 파일 이미지

client.upload_file(file_name, bucket, key) #파일 저장
