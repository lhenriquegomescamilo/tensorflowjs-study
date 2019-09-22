$('#result').text('');

function exibir(str = '') {
    $('#result').text(str);
}

function executar() {
    exibir('...processando.');
    let txt = '';

    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [1], units: 1})
        ]
    });
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError',
        metrics: ['accuracy']
    });

    const eixoX = tf.tensor([1, 2, 3, 4], [4, 1]);
    const eixoY = tf.tensor([[11], [22], [33], [44]]);
    const input = tf.tensor([5, 6, 7], [3, 1]);

    model.fit(eixoX, eixoY, {epochs: 100}).then(() => {
        const output = model.predict(input).dataSync().map(v => Math.ceil(v));
        const tensorOutput = tf.tensor(output);

        txt += 'Regressão Linear Simples com Rede Neural:\n';
        txt += 'TREINAMENTO:\n';
        txt += eixoY.flatten().toString() + '\n\n';
        txt += eixoY.flatten().toString() + '\n\n';
        txt += 'ENTRADA:\n';
        txt += input.flatten().toString() + '\n\n';
        txt += 'SAIDA:\n\n';
        txt += tensorOutput.toString();
        exibir(txt);

    });
}
