# Bilinear_CNN
A pytorch implementation of Bilinear CNNs for Fine-grained Visual Recognition(BCNN).


Step 1. Modify the path to the image in data.py.


Step 2. train the fc layer only. It gives 77.30% test set accuracy.
    	python train_last.py

Step 3. Fine-tune all layers. It gives 84.40% test set accuracy.
	python train_finetune.py
