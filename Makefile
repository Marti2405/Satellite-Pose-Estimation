.PHONY: default all clean

SPEED_URL=https://zenodo.org/records/5588480/files/speedplusv2.zip
NB_IMAGES?=10


default: all

all: speedplusv2 model/references

clean:
	rm -rf speedplusv2.zip

speedplusv2: 
	wget -O $@.zip -np ${SPEED_URL} \
	&& unzip $@.zip \
	&& rm -f $@.zip

dataset/images: dataset.blend dataset.py
	blender $< -F PNG --background --python dataset.py -- -c ${NB_IMAGES} -o $@

dataset/labels.json: concat.py dataset/images
	python $^ $@ \
	&& rm -f dataset/*.json

dataset/camera.json: dataset/images
	mv dataset/images/camera.json $@

dataset: dataset/images dataset/camera.json dataset/labels.json

transfo: model/key_points.json dataset/labels.json
	python transfo.py $^

model/references: model/search_ref.py speedplusv2
	python $^ $@