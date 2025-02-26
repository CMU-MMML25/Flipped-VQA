
cd pretrained

pip install llama-stack

llama model list --show-all

llama download --source meta --model-id  Llama-2-7b

mv ~/home/ubuntu/.llama/checkpoints/Llama-2-7b .
mv Llama-2-7b llama

cd llama
mkdir 7B
mv checklist.chk 7B/checklist.chk
mv consolidated.00.pth 7B/consolidated.00.pth
mv params.json 7B/params.json
