
const NSFWNET_WEIGHTS_PATH ='web_model/model.json';

const IMAGE_SIZE = 256;
const IMAGE_CROP_SIZE = 224;
const TOPK_PREDICTIONS = 5;

const NSFW_CLASSES = {
  0: 'drawing',
  1: 'hentai',
  2: 'neural',
  3: 'porn',
  4: 'sexy',
};


let nsfwnet;
const nsfwnetDemo = async () => {

  nsfwnet =await tf.loadGraphModel('web_model/model.json');


  nsfwnet.predict(tf.zeros([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])).dispose();

  console.log('Model Warm complete');

  const image_Element = document.getElementById('test_draw');
  if (image_Element.complete && image_Element.naturalHeight !== 0) {

    predict(image_Element);
    image_Element.style.display = 'block';
  } 
  document.getElementById('file-container').style.display = 'block';
};


async function predict(imgElement) {
  

  const logits = tf.tidy(() => {

    const img = tf.browser.fromPixels(imgElement).toFloat();
    const crop_image = tf.slice(img, [16, 16, 0], [224, 224, -1]);
    const img_reshape = tf.reverse(crop_image, [-1]);

    let imagenet_mean = tf.expandDims([103.94, 116.78, 123.68], 0);
    imagenet_mean = tf.expandDims(imagenet_mean, 0);

    const normalized = img_reshape.sub(imagenet_mean);

    const batched = normalized.reshape([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3]);

    return nsfwnet.predict(batched);
  });

  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  display(classes);
}


async function getTopKClasses(logits, topK){
  const values = await logits.data();
  sortArray = Array.from(values).map((value, index) => {
    return {
      value: value,
      index: index
    };
  }
  ).sort((a, b) => {
    return b.value - a.value;
  }).slice(0, topK);

  return sortArray.map(x => {
    return {
      className: NSFW_CLASSES[x.index],
      probability: x.value
    };
  }
  );
}




function display(classes){
  console.log(classes);
}



nsfwnetDemo();
