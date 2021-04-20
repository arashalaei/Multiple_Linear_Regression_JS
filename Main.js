/* jslint esversion: 9 */
/**
 * @author Arash Alaei <arashalaei22@gmail.com>
 * @since 02/00/2021
 */
require('@tensorflow/tfjs-node');
const dfd = require('danfojs-node'); // import pandas as pd
const MLR = require('ml-regression').MultivariateLinearRegression;
const tf = require('@tensorflow/tfjs'); // import numpy as np
const path = require("path");


(async() => {
    // Importing the dataset
    const dataset = await dfd.read_csv(`file://${path.join(__dirname,'./50_Startups.csv')}`);

    let x = dataset.iloc({rows:[':'], columns:[':4']}).values;
    let y = dataset.iloc({rows:[':'], columns:['4']}).values;
    
    // Encoding categorial feature
    let ohe = new dfd.OneHotEncoder();
    ohe.fit(dataset['State']);
    let sf_enc = ohe.transform(dataset['State'].values).data;
    x = fit_transform(x,sf_enc);

    // Spliting the dataset into the trainig and test set.
    let x_train = [], 
        x_test  = [], 
        y_train = [], 
        y_test  = [];


    let x_shuff = [...x];
    let y_shuff = [...y];
    shuffleCombo(x_shuff, y_shuff);

    x_train = [...x_shuff.slice(0, Math.floor(0.8 * x.length))];
    x_test  = [...x_shuff.slice(Math.floor(0.8 * x.length))];

    y_train = [...y_shuff.slice(0, Math.floor(0.8 * y.length))];
    y_test  = [...y_shuff.slice(Math.floor(0.8 * y.length))];

    const regressor = new MLR(x_train, y_train);
    y_pred =  tf.tensor(regressor.predict(x_test));
    y_test = tf.tensor(y_test);
    
    tf.concat([y_pred, y_test],1).print();
})();

function fit_transform(arr,enc){
    let temp = [];
    arr.forEach((item,index) => {
        item.pop();
        temp.push([...enc[index],...item]);
    });
    return temp;
}   


function shuffleCombo(array, array2) {
    
    if (array.length !== array2.length) {
      throw new Error(
        `Array sizes must match to be shuffled together ` +
        `First array length was ${array.length}` +
        `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let temp, temp2;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
      // Pick a random index
      index = (Math.random() * counter) | 0;
      // Decrease counter by 1
      counter--;
      // And swap the last element of each array with it
      temp = array[counter];
      temp2 = array2[counter];
      array[counter] = array[index];
      array2[counter] = array2[index];
      array[index] = temp;
      array2[index] = temp2;
    }
  }
  