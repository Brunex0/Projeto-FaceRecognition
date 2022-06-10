# Recognition of individuals in video surveillance environments

## Configuration and execution
### Datasets
First you need to download the Casia-Webface for trainning and LFW-funneled for validation and testing and icbrw to test in surveillance environments.
* Casia-Webface: https://paperswithcode.com/dataset/casia-webface
* LFW-funneled: http://vis-www.cs.umass.edu/lfw/
* ICBRW: http://icbrw.di.ubi.pt/

### Face Alignment
To obtain better results, we need to align the faces.
You can align all datasets using `align_dataset_mtcnn.py` (from https://github.com/davidsandberg/facenet).
Unfortunately, the tensorflow version was outdated so the following minor changes had to be made in `align_dataset_mtcnn.py`:
```
remove the value, feed_in, dim = (inp, input_shape[-1])

#need to import the following packages
import tensorflow.compat.v1 as tf #ignore the red warning
import imageio # to replace misc.imread() as format: imageio.imread(os.path.join(emb_dir, i))
from PIL import Image # to replace misc.resize as format: scaled = np.array(Image.fromarray(cropped).resize((args.image_size, args.image_size), Image.BILINEAR))
#add this things to the code
```
In the script `detect_face.py` change:
```
Row 85, replace as format: data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()
```

Finally, you need to change the path of the directories in `run_align.sh` and then run the script.

### Split the train dataset in train and test
At this stage, you need to split the training dataset in 80% of the photos to train and 20% for validation. You can use `create_train_test_splits.py` to do that.

### The config file
The `config.yml` file contains all the configurations necessary to train, test, prepare the data, to create the pairs for the icbrw and the paths to use to do evaluation.

### Train
To train the model you need to change the following paths according to the location of your files:
* train-path;
* validation-path.

If you want to run the custom callback for lfw in the end of each epoch change the following parameters:
* lfw-callback;
* lfw-callback-pairs.

If you don´t want to do this custom callback you have to change the following lines in `train.py`.
```
Comment line 81
Change callbacks at line 88 to: callbacks=[modelCheckpoint_callback]
```

Finally, it's possible to run the file as `python3 train.py`

### Other notes
* **Baseline**: Train with no align dataset and test with no align dataset and without L2 normalization;
* **Align**: Train with align dataset and test with align dataset
* **Align-L2**: Train with align dataset. Test with align dataset, using L2 normalization for the features and calculate euclidean distance between the features
* **Align-L2-CosineSim**: Train with align dataset. Test with align dataset, using L2 normalization for the features and calculate cosine similarity.