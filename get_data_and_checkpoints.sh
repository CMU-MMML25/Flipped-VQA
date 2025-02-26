sudo apt update
sudo apt install git-lfs

git lfs install
git clone https://huggingface.co/datasets/ikodoh/Flipped-VQA-Data

mv ./Flipped-VQA-Data/data ./
mv ./Flipped-VQA-Data/checkpoint ./

unzip ./data/tvqa/tvqa_subtitles.zip -d ./data/tvqa
rm -rf Flipped-VQA-Data ./data/tvqa/tvqa_subtitles.zip
