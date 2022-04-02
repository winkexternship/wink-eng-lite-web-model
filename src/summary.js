// NON BOW PREPROCESSING FUNCTIONS

let preprocessingNonBow = function ( rdd, its ) {

    // required information
    const numOfSentences = rdd.sentences.length;
    const sentences = rdd.sentences;
    const tokens = rdd.tokens;
    const cache = rdd.cache;
    const aptTokens = [];
    const paraStarts = [ 0 ];
    const aptPOS =  [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

    let para = [];
    // generation of aptTokens
    for ( let i = 0; i < numOfSentences; i += 1 ) {
      const sen = [];
      for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1 ) {
        if (its.type( j, tokens, cache ) === 'tabCRLF' ) {
          aptTokens.push(para)
          paraStarts.push(i);
          para = []
        }
        if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
          sen.push(its.normal( j, tokens, cache ));
        }
      }
      para.push(sen);
    }

    aptTokens.push(para);
    const textInfo = { aptTokens: aptTokens, paraStarts: paraStarts };

    return textInfo;

};

let modifiedPreProcessingNonBow = function ( rdd, its) {

  // required information
  const numOfSentences = rdd.sentences.length;
  const sentences = rdd.sentences;
  const tokens = rdd.tokens;
  const cache = rdd.cache;
  const aptTokens = [];
  const paraStarts = [ 0 ];
  const aptPOS =  [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

  // generation of aptTokens
  let sentenceCount = 0;
  let para = [];
  for ( let i = 0; i < numOfSentences; i += 1 ) {
    const sen = [];
    for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1 ) {
      if (its.type( j, tokens, cache ) === 'tabCRLF' && sentenceCount > 6) {
        aptTokens.push(para);
        sentenceCount = 0;
        paraStarts.push(i);
        para = [];
      } else if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
        sen.push(its.normal( j, tokens, cache ));
      }
    }
    sentenceCount += 1;
    para.push(sen);
  }
  aptTokens.push(para);
  const textInfo = { aptTokens: aptTokens, paraStarts: paraStarts };
  return textInfo;

};

let wholePreProcessingNonBow = function ( rdd, its, as ) {

  // required information
  const numOfSentences = rdd.sentences.length;
  const sentences = rdd.sentences;
  const tokens = rdd.tokens;
  const cache = rdd.cache;
  const aptTokens = [];
  const paraStarts = [ 0 ];
  const aptPOS =  [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

  // generation of aptTokens
  let para = [];
  for ( let i = 0; i < numOfSentences; i += 1 ) {
    const sen = [];
    for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1 ) {
      if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
        sen.push(its.normal( j, tokens, cache ));
      }
    }
    para.push(sen);
  }
  aptTokens.push(para);
  const textInfo = { aptTokens: aptTokens, paraStarts: paraStarts };
  return textInfo;

};

// BOW PROCESSING FUNCTIONS

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
          bow.push(para);
          paraStarts.push(i);
          para = [];
        }
        if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
          sen.push(its.normal( j, tokens, cache ));
        }
      }
      para.push(as.bow(sen));
    }

    bow.push(para);
    const textInfo = { bow: para, paraStarts: paraStarts };
    return textInfo;

};

// BM25 PROCESSING FUNCTIONS

let bm25Bow = function ( aptTokens, BM25Vectorizer, its ) {

    // variables
    const numOfParagraphs = aptTokens.length;
    const bow = [];

    for ( let i = 0; i < numOfParagraphs; i += 1 ) {
      const bm25 = BM25Vectorizer();
      aptTokens[i].forEach( (colToken) => bm25.learn(colToken) );
      const para = [];
      for ( let j = 0; j < aptTokens[i].length; j += 1 ) {
        para.push(bm25.doc(j).out(its.bow));
      }
      bow.push(para);
    }

    return bow;

};

// GRAPH CREATION FUNCTIONS

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

// RANKING FUNCTIONS

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

// MAIN FUNCTIONS

// // Comment out blocks for which you want to run tests
// let summary = function ( rdd, its, as, simmilarity, BM25Vectorizer ) {

//   const outer = [];
//   let inner = [];
//   for ( let i = 0; i < paraStarts.length - 1; i += 1 ) {
//     for ( let j = 0; j < weights.length; j += 1 ) {
//       if ( weights[ j ].idx >= paraStarts[ i ] && weights[ j ].idx < paraStarts[ i + 1 ])
//         inner.push( weights[ j ] );
//     }
//     outer.push( inner );
//     inner = [];
//   }
//   for ( let i = 0; i < weights.length; i += 1) {
//     if ( weights[ i ].idx >= paraStarts[ paraStarts.length - 1 ] )
//       inner.push( weights[ i ] );
//   }
//   outer.push( inner );
//   return outer;

// };

// Comment out blocks for which you want to run tests
let summary = function ( rdd, its, as, simmilarity, BM25Vectorizer ) {

    const weights = [];
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
    // const bow = bm25Bow( textInfo.aptTokens, BM25Vectorizer, its );
    // weights =  pagerankWithoutWeights( createGraphCosine( bow[i], simmilarity ) ) ;
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // Common Intersection + Pagerank With Weights
    // const textInfo = preprocessingNonBow( rdd, its);
    // weights = pagerankWithWeights( createGraphCommonTokens( textInfo.aptTokens ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // Cosine Simmilarity + Pagerank With Weights
    // const textInfo = preprocessingBow( rdd, its, as );
    // for ( let i = 0; i < textInfo.bow.length; i += 1 ) {
    //     weights.push( pagerankWithWeights( createGraphCosine( textInfo.bow[i], simmilarity ) ) );
    // }
    // summaryInfo.weights = weights;
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // // BM25 + Pagerank With Weights
    // const textInfo = preprocessingNonBow( rdd, its);
    // const bow = bm25Bow( textInfo.aptTokens, BM25Vectorizer, its );
    // weights = pagerankWithWeights( createGraphCosine( bow[i], simmilarity ) );
    // summaryInfo.weights = weightToPara(weights, textInfo.paraStarts);
    // summaryInfo.paraStarts = textInfo.paraStarts;

    // BM25 Whole Document
    const textInfo = wholePreProcessingNonBow( rdd, its, as );
    const bow = bm25Bow( textInfo.aptTokens, BM25Vectorizer, its );
    for ( let i = 0; i < textInfo.aptTokens.length; i += 1 ) {
        weights.push( pagerankWithWeights( createGraphCosine( bow[i], simmilarity ) ) );
    }
    summaryInfo.weights = weights;
    summaryInfo.paraStarts = textInfo.paraStarts;

    // // BM25 Combined Para Logic
    // const textInfo = modifiedPreProcessingNonBow( rdd, its, as);
    // const bow = bm25Bow( textInfo.aptTokens, BM25Vectorizer, its );
    // for ( let i = 0; i < textInfo.aptTokens.length; i += 1 ) {
    //   weights.push( pagerankWithWeights( createGraphCosine( bow[i], simmilarity ) ) );
    // }
    // summaryInfo.weights = weights;
    // summaryInfo.paraStarts = textInfo.paraStarts;

    return summaryInfo;

};

module.exports = summary;
