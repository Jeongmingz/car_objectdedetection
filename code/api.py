from roboflow import Roboflow
import json
import numpy as np

with open('data/data.json', 'r') as file:
	data = json.load(file)

if data == '':
	rf = Roboflow(api_key="CFD5OGXqiYxYtTQoEBGo")
	project = rf.workspace().project("car-models-08rnv")
	model = project.version(4).model

	# infer on a local image
	data = model.predict("test.png", confidence=40, overlap=30).json()
	print(data)

	# visualize your prediction
	model.predict("test.png", confidence=40, overlap=30).save("prediction.jpg")

	with open('data/data.json', 'w') as file:
		json.dump(data, file)


with open('coordinate/img_labeling_list.json', 'r') as file:
	coordinate = json.load(file)

is_parked = np.zeros(24, dtype="int")
for i, datas in enumerate(data.get("predictions")):
	x_coordinate = datas.get('x')
	y_coordinate = datas.get('y')
	for i, coordinates in enumerate(coordinate):
		x_list, y_list = coordinates.get('x'), coordinates.get('y')

		if x_list[0] <= x_coordinate <= x_list[1]:
			if y_list[0] <= y_coordinate <= y_list[1]:
				is_parked[coordinates.get('id')] = True

print(is_parked)
dict_ = {"parkingInfo": is_parked.tolist()}
with open('result/result.json', 'w') as file:
	json.dump(dict_, file)
