let preprocessingNonBow = function ( rdd, its ) {

    // required information
    const numOfSentences = rdd.sentences.length;
    const sentences = rdd.sentences;
    const tokens = rdd.tokens;
    const cache = rdd.cache;
    const aptTokens = [];
    const paraStarts = [ 0 ];
    const aptPOS =  [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

    // generation of aptTokens
    for ( let i = 0; i < numOfSentences; i += 1 ) {
      const sen = [];
      for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1 ) {
        if (its.type( j, tokens, cache ) === 'tabCRLF' ) {
          paraStarts.push(i);
        }
        if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
          sen.push(its.normal( j, tokens, cache ));
        }
      }
      aptTokens.push(sen);
    }
    const textInfo = { aptTokens: aptTokens, paraStarts: paraStarts };

    return textInfo;

};


let preprocessingBow = function ( rdd, its, as ) {

    // required information
    const numOfSentences = rdd.sentences.length;
    const sentences = rdd.sentences;
    const tokens = rdd.tokens;
    const cache = rdd.cache;
    const bow = [];
    const paraStarts = [ 0 ];
    const aptPOS =  [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

    // generation of aptTokens
    let para = [];
    for ( let i = 0; i < numOfSentences; i += 1 ) {
      const sen = [];
      for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1 ) {
        if (its.type( j, tokens, cache ) === 'tabCRLF' ) {
          paraStarts.push(i);
        }
        if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
          sen.push(its.normal( j, tokens, cache ));
        }
      }
      para.push(as.bow(sen));
    }
    const textInfo = { bow: para, paraStarts: paraStarts };
    return textInfo;

};

let bm25Bow = function ( aptTokens, bm25, its ) {

    // variables
    aptTokens.forEach( (colToken) => bm25.learn(colToken) );
    const bow = [];
    for ( let j = 0; j < aptTokens.length; j += 1 ) {
      bow.push(bm25.doc(j).out(its.bow));
    }
    return bow;

};

let createGraphCommonTokens = function ( para ) {

    const numOfSentences = para.length;
    const senGraph = new Array(numOfSentences);

    for ( let i = 0; i < numOfSentences; i += 1) {
      senGraph[i] = new Array(numOfSentences);
    }

    for ( let i = 0; i < numOfSentences; i += 1) {
      senGraph[i][i] = 0;
        for ( let j = i + 1; j < numOfSentences; j += 1) {
          const numOfCommonTokens = para[i].filter((token) => para[j].includes(token)).length;
          senGraph[i][j] = numOfCommonTokens / (Math.log(para[i].length) + Math.log(para[j].length));
          if ( !Number.isFinite(senGraph[i][j] )  )
            senGraph[i][j] = 0;
          senGraph[j][i] = senGraph[i][j];
        }
    }
    return senGraph;

};

let createGraphCosine = function ( paraBow, simmilarity ) {

    const numOfSentences = paraBow.length;
    const senGraph = new Array(numOfSentences);

    for ( let i = 0; i < numOfSentences; i += 1) {
      senGraph[i] = new Array(numOfSentences);
    }

    for ( let i = 0; i < numOfSentences; i += 1) {
      senGraph[i][i] = 0;
        for ( let j = i + 1; j < numOfSentences; j += 1) {
          senGraph[i][j] = simmilarity.bow.cosine(paraBow[i], paraBow[j]);
          if ( !Number.isFinite(senGraph[i][j] )  )
            senGraph[i][j] = 0;
          senGraph[j][i] = senGraph[i][j];
        }
    }
    return senGraph;

};

let pagerankWithoutWeights = function ( paraSenGraph ) {

    const numOfSentences = paraSenGraph.length;
    const nonZero = [];
    const ranks = [];

    for ( let i = 0; i < numOfSentences; i += 1) {
      nonZero[i] = paraSenGraph[i].reduce((previousValue) =>  previousValue + 1 ,0);
    }

    for ( let i = 0; i < numOfSentences; i += 1 ) {
      const s = paraSenGraph[i].reduce((previousValue, currentValue, idx) =>  previousValue + ( currentValue / nonZero[idx] )  ,0);
      ranks.push({ idx: i, val: s });
    }
    ranks.sort( (a, b) => b.val - a.val );
    return ranks;

};

let pagerankWithWeights = function ( paraSenGraph ) {

    const numOfSentences = paraSenGraph.length;
    const colSum = [];
    const weights = [];
    const maxIteration = 100;
    const dampingFactor = 0.85;
    const convergenceThreshold =  0.0001;
    let numOfConvergence = 0;

    for ( let i = 0; i < numOfSentences; i += 1) {
      colSum[i] = paraSenGraph[i].reduce((previousValue, currentValue) =>  previousValue + currentValue ,0);
    }

    for ( let i = 0; i < numOfSentences; i += 1) {
      weights.push({ idx: i, val: 1 / numOfSentences });
    }

    for ( let i = 0; i < maxIteration; i += 1) {
      for ( let j = 0; j < numOfSentences; j += 1) {
        let rank = 1 - dampingFactor;
        rank += dampingFactor * paraSenGraph[j].reduce((previousValue, currentValue, idx) => previousValue + ((currentValue / colSum[idx]) * weights[idx].val) ,0);
        if ( Math.abs(weights[j] - rank) <= convergenceThreshold ) {
          numOfConvergence += 1;
        }
        weights[j].val = rank;
      }
      if ( numOfConvergence === numOfSentences ) {
        break;
      }
    }
    weights.sort( (a, b) => b.val - a.val );
    return weights;
};

let weightToPara = function(weights, paraStarts) {

  const outer = [];
  let inner = [];
  for ( let i = 0; i < paraStarts.length - 1; i += 1 ) {
    for ( let j = 0; j < weights.length; j += 1 ) {
      if ( weights[ j ].idx >= paraStarts[ i ] && weights[ j ].idx < paraStarts[ i + 1 ])
        inner.push( weights[ j ] );
    }
    outer.push( inner );
    inner = [];
  }
  for ( let i = 0; i < weights.length; i += 1) {
    if ( weights[ i ].idx >= paraStarts[ paraStarts.length - 1 ] )
      inner.push( weights[ i ] );
  }
  outer.push( inner );
  return outer;

};

// Comment out blocks for which you want to run tests
let summary = function ( rdd, its, as, simmilarity, bm25 ) {

    let weights = [];
    const summaryInfo = {};

    // Common Intersection + Pagerank Without Weights
    // const textInfo = preprocessingNonBow( rdd, its);
    // weights = pagerankWithoutWeights( createGraphCommonTokens( textInfo.aptTokens ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // Cosine Simmilarity + Pagerank Without Weights
    // const textInfo = preprocessingBow( rdd, its, as );
    // weights =  pagerankWithoutWeights( createGraphCosine( textInfo.bow, simmilarity ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // BM25 + Pagerank Without Weights
    // const textInfo = preprocessingNonBow( rdd, its);
    // const bow = bm25Bow( textInfo.aptTokens, bm25, its );
    // weights =  pagerankWithoutWeights( createGraphCosine( bow, simmilarity ) ) ;
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // Common Intersection + Pagerank With Weights
    // const textInfo = preprocessingNonBow( rdd, its);
    // weights = pagerankWithWeights( createGraphCommonTokens( textInfo.aptTokens ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // Cosine Simmilarity + Pagerank With Weights
    // const textInfo = preprocessingBow( rdd, its, as );
    // weights = pagerankWithWeights( createGraphCosine( textInfo.bow, simmilarity ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // // BM25 + Pagerank With Weights
    const textInfo = preprocessingNonBow( rdd, its);
    const bow = bm25Bow( textInfo.aptTokens, bm25, its );
    weights = pagerankWithWeights( createGraphCosine( bow, simmilarity ) );
    summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    summaryInfo.paraStarts = textInfo.paraStarts;

    return summaryInfo;

};

module.exports = summary;
