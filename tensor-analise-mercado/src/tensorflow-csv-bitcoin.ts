import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { Logger } from '@nestjs/common';
import * as R from 'ramda';

const fnNoEmpty = R.compose(R.not, R.empty);
const logger = new Logger('tensorflow-csv-dolar');
const pathCSV = path.join(__dirname, '../assets/csv/dado-historico-bitcoin-janeiro-setembro-treinamento.csv');
const file = fs.readFileSync(pathCSV, { encoding: 'utf8' });
const fileCleaned = file.toString().trim();
const linhas = fileCleaned.split('\r\n');
const eixoX = [];
const eixoY = [];
const removeQuote = (stringValue) => {
  if (!stringValue) {
    return stringValue;
  }
  return stringValue.replace('"', '').replace('"', '');
};
let quantidadeLinhas = 0;
for (let indice = 1; indice < linhas.length; indice++) {
  try {
    let celula1 = [];
    const lineCelular1 = linhas[indice + 1].split(',');
    if (lineCelular1 && lineCelular1.every(fnNoEmpty)) {
      celula1 = lineCelular1;
    }
    const celula2 = linhas[indice].split(',');
    const FechamentoX = parseFloat(removeQuote(celula1[1]));
    const AberturaX = parseFloat(removeQuote(celula1[2]));
    const MaximaX = parseFloat(removeQuote(celula1[3]));
    const MinimaX = parseFloat(removeQuote(celula1[4]));
    const dadosEixoX = [FechamentoX, AberturaX, MaximaX, MinimaX];

    eixoX.push(dadosEixoX);
    logger.log(dadosEixoX, 'Eixo X' + indice);

    const FechamentoY = parseFloat(removeQuote(celula2[1]));
    const AberturaY = parseFloat(removeQuote(celula2[2]));
    const MaximaY = parseFloat(removeQuote(celula2[3]));
    const MinimaY = parseFloat(removeQuote(celula2[4]));
    const dadoEixoY = [FechamentoY, AberturaY, MaximaY, MinimaY];
    eixoY.push(dadoEixoY);

    logger.log(dadoEixoY, 'eixo Y');
    quantidadeLinhas++;

  } catch (error) {
    logger.error(error, `erro em ${indice}`);
  }
}

const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [4], units: 4}),
  ],
});
// const inputLayer = tf.layers.dense({ units: 4, inputShape: [4] });
// model.add(inputLayer);
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
  metrics: ['accuracy'],
});
const x = tf.tensor(eixoX, [quantidadeLinhas, 4]);
const y = tf.tensor(eixoY);

const arrInput = [[10.198, 10.303, 10.340, 10.105]]; // "20.09.2019"
// const arrInput = [[10.013, 10.197, 10.199, 9.958]]; // "21.09.2019",

const input = tf.tensor(arrInput, [1, 4]);

model.fit(x, y, { epochs: 600 }).then(() => {
  // @ts-ignore
  const output = model.predict(input).dataSync();
  logger.log('PRECO DAS COTACOES BITCOIN');
  logger.log(`Fechamento: \t$ ${Number(output[0].toFixed(3))}`);
  logger.log(`Abertura:   \t$ ${Number(output[1].toFixed(3))}`);
  logger.log(`Maximo:     \t$ ${Number(output[2].toFixed(3))}`);
  logger.log(`Minimo:     \tR$ ${Number(output[3].toFixed(3))}`);
});
// }).catch(error => logger.error(error));
