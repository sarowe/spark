/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature;

import com.google.common.collect.Lists;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Row;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

import java.util.List;

public class JavaLuceneAnalyzerSuite {
  private transient JavaSparkContext jsc;
  private transient SQLContext jsql;

  @Before
  public void setUp() {
    jsc = new JavaSparkContext("local", "JavaLuceneAnalyzerSuite");
    jsql = new SQLContext(jsc);
  }

  @After
  public void tearDown() {
    jsc.stop();
    jsc = null;
  }

  @Test
  public void testStandardTokenizer() {
    LuceneAnalyzer analyzer1 = new LuceneAnalyzer()
        .setInputCol("rawText")
        .setOutputCol("tokens"); // Default analysis schema: StandardTokenizer + LowerCaseFilter

    assertExpectedTokens(analyzer1, Lists.newArrayList(
        new TokenizerTestData("Test for tokenization.", new String[]{"test", "for", "tokenization"}),
        new TokenizerTestData("Te,st. punct", new String[]{"te", "st", "punct"})));

    assertExpectedTokens(analyzer1, Lists.newArrayList(
        new TokenizerTestData("Test for tokenization.", new String[]{"test", "for", "tokenization"}),
        new TokenizerTestData("Te,st. punct", new String[]{"te", "st", "punct"})));

    String analysisSchema1 = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'analyzers': [{\n" +
        "  'name': 'StdTok_max10',\n" +
        "  'tokenizer': {\n" +
        "    'type': 'standard',\n" +
        "    'maxTokenLength': '10'\n" +
        "  }\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'regex': '.+',\n" +
        "  'analyzer': 'StdTok_max10'\n" +
        "}]}\n");
    analyzer1.setAnalysisSchema(analysisSchema1);
    assertExpectedTokens(analyzer1, Lists.newArrayList(
        new TokenizerTestData("我是中国人。 １２３４ Ｔｅｓｔｓ ",
            new String[]{"我", "是", "中", "国", "人", "１２３４", "Ｔｅｓｔｓ"}),
        new TokenizerTestData("some-dashed-phrase", new String[]{"some", "dashed", "phrase"})));

    String analysisSchema2 = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'defaultLuceneMatchVersion': '4.10.4',\n" +
        "'analyzers': [{\n" +
        "  'name': 'StdTok_max3',\n" +
        "  'tokenizer': {\n" +
        "    'type': 'standard',\n" +
        "    'maxTokenLength': '3'\n" +
        "  }\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'regex': '.+',\n" +
        "  'analyzer': 'StdTok_max3'\n" +
        "}]}\n");
    LuceneAnalyzer analyzer2 = new LuceneAnalyzer()
        .setAnalysisSchema(analysisSchema2)
        .setInputCol("rawText")
        .setOutputCol("tokens");
    assertExpectedTokens(analyzer2, Lists.newArrayList(
        new TokenizerTestData("Test for tokenization.",
            new String[]{"Tes", "t", "for", "tok", "eni", "zat", "ion"}),
        new TokenizerTestData("Te,st.  punct", new String[]{"Te", "st", "pun", "ct"})));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testCharFilters() {
    String analysisSchema1 = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'analyzers': [{\n" +
        "  'name': 'strip_alpha_std_tok',\n" +
        "  'charFilters': [{\n" +
        "    'type': 'patternreplace',\n" +
        "    'pattern': '[A-Za-z]+',\n" +
        "    'replacement': ''\n" +
        "  }],\n" +
        "  'tokenizer': {\n" +
        "    'type': 'standard'\n" +
        "  }\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'regex': '.+',\n" +
        "  'analyzer': 'strip_alpha_std_tok'\n" +
        "}]}\n");
    LuceneAnalyzer analyzer = new LuceneAnalyzer()
        .setAnalysisSchema(analysisSchema1)
        .setInputCol("rawText")
        .setOutputCol("tokens");

    assertExpectedTokens(analyzer, Lists.newArrayList(
        new TokenizerTestData("Test for 9983, tokenization.", new String[]{"9983"}),
        new TokenizerTestData("Te,st. punct", new String[]{})));

    String analysisSchema2 = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'analyzers': [{\n" +
        "  'name': 'htmlstrip_drop_removeme_std_tok',\n" +
        "  'charFilters': [{\n" +
        "      'type': 'htmlstrip'\n" +
        "    }, {\n" +
        "      'type': 'patternreplace',\n" +
        "      'pattern': 'removeme',\n" +
        "      'replacement': ''\n" +
        "  }],\n" +
        "  'tokenizer': {\n" +
        "    'type': 'standard'\n" +
        "  }\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'name': 'rawText',\n" +
        "  'analyzer': 'htmlstrip_drop_removeme_std_tok'\n" +
        "}]}\n");
    analyzer.setAnalysisSchema(analysisSchema2);

    assertExpectedTokens(analyzer, Lists.newArrayList(
        new TokenizerTestData("<html><body>remove<b>me</b> but leave<div>the&nbsp;rest.</div></body></html>",
            new String[]{"but", "leave", "the", "rest"})));
  }

  @Test
  public void testTokenFilters() {
    String analysisSchema = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'analyzers': [{\n" +
        "  'name': 'std_tok_possessive_stop_lower',\n" +
        "  'tokenizer': {\n" +
        "    'type': 'standard'\n" +
        "  },\n" +
        "  'filters': [{\n" +
        "      'type': 'englishpossessive'\n" +
        "    }, {\n" +
        "      'type': 'stop',\n" +
        "      'ignoreCase': 'true',\n" +
        "      'format': 'snowball',\n" +
        "      'words': 'org/apache/lucene/analysis/snowball/english_stop.txt'\n" +
        "    }, {\n" +
        "      'type': 'lowercase'\n" +
        "  }]\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'name': 'rawText',\n" +
        "  'analyzer': 'std_tok_possessive_stop_lower'\n" +
        "}]}\n");
    LuceneAnalyzer analyzer = new LuceneAnalyzer()
        .setAnalysisSchema(analysisSchema)
        .setInputCol("rawText")
        .setOutputCol("tokens");
    assertExpectedTokens(analyzer, Lists.newArrayList(
        new TokenizerTestData("Harold's not around.", new String[]{"harold", "around"}),
        new TokenizerTestData("The dog's nose KNOWS!", new String[]{"dog", "nose", "knows"})));
  }

  @Test
  public void testUAX29URLEmailTokenizer() {
    String analysisSchema = json("{\n" +
        "'schemaType': 'LuceneAnalyzerSchema.v1',\n" +
        "'analyzers': [{\n" +
        "  'name': 'uax29urlemail_2000',\n" +
        "  'tokenizer': {\n" +
        "    'type': 'uax29urlemail',\n" +
        "    'maxTokenLength': '2000'\n" +
        "  }\n" +
        "}],\n" +
        "'inputColumns': [{\n" +
        "  'regex': '.+',\n" +
        "  'analyzer': 'uax29urlemail_2000'\n" +
        "}]}\n");
    LuceneAnalyzer analyzer = new LuceneAnalyzer()
        .setAnalysisSchema(analysisSchema)
        .setInputCol("rawText")
        .setOutputCol("tokens");
    assertExpectedTokens(analyzer, Lists.newArrayList(
        new TokenizerTestData("Click on https://www.google.com/#q=spark+lucene",
            new String[]{"Click", "on", "https://www.google.com/#q=spark+lucene"}),
        new TokenizerTestData("Email caffeine@coffee.biz for tips on staying@alert",
            new String[]{"Email", "caffeine@coffee.biz", "for", "tips", "on", "staying", "alert"})));
  }

  private void assertExpectedTokens(LuceneAnalyzer analyzer, List<TokenizerTestData> testData) {
    JavaRDD<TokenizerTestData> rdd = jsc.parallelize(testData);
    Row[] pairs = analyzer.transform(jsql.createDataFrame(rdd, TokenizerTestData.class))
        .select("tokens", "wantedTokens")
        .collect();
    for (Row r : pairs) {
      Assert.assertEquals(r.get(0), r.get(1));
    }
  }

  private String json(String singleQuoted) {
    return singleQuoted.replaceAll("'", "\"");
  }
}
