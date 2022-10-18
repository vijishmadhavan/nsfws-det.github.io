const NSFWNET_WEIGHTS_PATH ='model/model.json';

const IMAGE_SIZE = 256;
const IMAGE_CROP_SIZE = 299;
const TOPK_PREDICTIONS = 5;

const NSFW_CLASSES = {
  0: 'drawing',
  1: 'hentai',
  2: 'neutral',
  3: 'porn',
  4: 'sexy',
};


let nsfwnet;
const nsfwnetDemo = async () => {

  nsfwnet =await tf.loadLayersModel('model/model.json');


  nsfwnet.predict(tf.zeros([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])).dispose();

  console.log('Model Warm complete');

  const image_Element = document.getElementById('test_draw');
  if (image_Element.complete && image_Element.naturalHeight !== 0) {

    predict(image_Element);
    image_Element.style.display = 'block';
  } 
  document.getElementById('test_draw').addEventListener('load', () => {
    predict(image_Element);
    image_Element.style.display = 'block';
  });
};




async function predict(imgElement) {
  

  const logits = tf.tidy(() => {

    const img = tf.browser.fromPixels(imgElement).toFloat();
    const offset = tf.scalar(127.5);
    const normalized = img.sub(offset).div(offset);
    const resized = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE]);
    const cropped = tf.image.resizeBilinear(resized, [IMAGE_CROP_SIZE, IMAGE_CROP_SIZE]);
    const batched = cropped.expandDims(0);
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
