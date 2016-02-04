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

package org.apache.spark.ml.feature

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{Row, DataFrame}

class LuceneAnalyzerSuite extends SparkFunSuite with MLlibTestSparkContext {
  import org.apache.spark.ml.feature.LuceneAnalyzerSuite._

  test("params") {
    ParamsSuite.checkParams(new LuceneAnalyzer)
  }

  test("StandardTokenizer") {
    val analyzer1 = new LuceneAnalyzer()
      .setInputCol("rawText")
      .setOutputCol("tokens")  // Default analysis schema: StandardTokenizer + LowerCaseFilter

    val dataset1 = sqlContext.createDataFrame(Seq(
      TokenizerTestData("Test for tokenization.", Array("test", "for", "tokenization")),
      TokenizerTestData("Te,st. punct", Array("te", "st", "punct"))
    ))
    testLuceneAnalyzer(analyzer1, dataset1)

    val dataset2 = sqlContext.createDataFrame(Seq(
      TokenizerTestData("我是中国人。 １２３４ Ｔｅｓｔｓ ",
        Array("我", "是", "中", "国", "人", "１２３４", "Ｔｅｓｔｓ")),
      TokenizerTestData("some-dashed-phrase", Array("some", "dashed", "phrase"))
    ))
    val analyzerConfig2 = """
                             |{
                             |  "schemaType": "LuceneAnalyzerSchema.v1",
                             |  "analyzers": [{
                             |    "name": "StdTok_max3",
                             |    "tokenizer": {
                             |      "type": "standard",
                             |      "maxTokenLength": "10"
                             |    }
                             |  }],
                             |  "inputColumns": [{
                             |    "name": "rawText",
                             |    "analyzer": "StdTok_max3"
                             |  }]
                             |}""".stripMargin
    analyzer1.setAnalysisSchema(analyzerConfig2)
    testLuceneAnalyzer(analyzer1, dataset2)

    val analyzerConfig3 = """
                            |{
                            |  "schemaType": "LuceneAnalyzerSchema.v1",
                            |  "defaultLuceneMatchVersion": "4.10.4",
                            |  "analyzers": [{
                            |    "name": "StdTok_max3",
                            |    "tokenizer": {
                            |      "type": "standard",
                            |      "maxTokenLength": "3"
                            |    }
                            |  }],
                            |  "inputColumns": [{
                            |    "regex": ".+",
                            |    "analyzer": "StdTok_max3"
                            |  }]
                            |}""".stripMargin
    val analyzer2 = new LuceneAnalyzer()
      .setAnalysisSchema(analyzerConfig3)
      .setInputCol("rawText")
      .setOutputCol("tokens")
    val dataset3 = sqlContext.createDataFrame(Seq(
      TokenizerTestData("Test for tokenization.",
        Array("Tes", "t", "for", "tok", "eni", "zat", "ion")),
      TokenizerTestData("Te,st.  punct", Array("Te", "st", "pun", "ct"))
    ))
    testLuceneAnalyzer(analyzer2, dataset3)
  }

  test("CharFilters") {
    val analyzerConfig1 = """
                            |{
                            |  "schemaType": "LuceneAnalyzerSchema.v1",
                            |  "analyzers": [{
                            |    "name": "strip_alpha_std_tok",
                            |    "charFilters":[{
                            |      "type": "patternreplace",
                            |      "pattern": "[A-Za-z]+",
                            |      "replacement": ""
                            |    }],
                            |    "tokenizer": {
                            |      "type": "standard"
                            |    }
                            |  }],
                            |  "inputColumns": [{
                            |    "regex": ".+",
                            |    "analyzer": "strip_alpha_std_tok"
                            |  }]
                            |}""".stripMargin
    val analyzer = new LuceneAnalyzer()
      .setAnalysisSchema(analyzerConfig1)
      .setInputCol("rawText")
      .setOutputCol("tokens")
    val dataset1 = sqlContext.createDataFrame(Seq(
      TokenizerTestData("Test for 9983, tokenization.", Array("9983")),
      TokenizerTestData("Te,st. punct", Array())
    ))
    testLuceneAnalyzer(analyzer, dataset1)

    val analyzerConfig2 = """
                            |{
                            |  "schemaType": "LuceneAnalyzerSchema.v1",
                            |  "analyzers": [{
                            |    "name": "htmlstrip_drop_removeme_std_tok",
                            |    "charFilters":[{
                            |        "type": "htmlstrip"
                            |      }, {
                            |        "type": "patternreplace",
                            |        "pattern": "removeme",
                            |        "replacement": ""
                            |    }],
                            |    "tokenizer": {
                            |      "type": "standard"
                            |    }
                            |  }],
                            |  "inputColumns": [{
                            |    "name": "rawText",
                            |    "analyzer": "htmlstrip_drop_removeme_std_tok"
                            |  }]
                            |}""".stripMargin
    analyzer.setAnalysisSchema(analyzerConfig2)
    val dataset2 = sqlContext.createDataFrame(Seq(
      TokenizerTestData(
        "<html><body>remove<b>me</b> but leave<div>the&nbsp;rest.</div></body></html>",
        Array("but", "leave", "the", "rest"))
    ))
    testLuceneAnalyzer(analyzer, dataset2)
  }

  test("TokenFilters") {
    val analyzerConfig = """
                           |{
                           |  "schemaType": "LuceneAnalyzerSchema.v1",
                           |  "analyzers": [{
                           |    "name": "std_tok_possessive_stop_lower",
                           |    "tokenizer": {
                           |      "type": "standard"
                           |    },
                           |    "filters":[{
                           |        "type": "englishpossessive"
                           |      }, {
                           |        "type": "stop",
                           |        "ignoreCase": "true",
                           |        "format": "snowball",
                           |        "words": "org/apache/lucene/analysis/snowball/english_stop.txt"
                           |      }, {
                           |        "type": "lowercase"
                           |    }]
                           |  }],
                           |  "inputColumns": [{
                           |    "regex": ".+",
                           |    "analyzer": "std_tok_possessive_stop_lower"
                           |  }]
                           |}""".stripMargin
    val analyzer = new LuceneAnalyzer()
      .setAnalysisSchema(analyzerConfig)
      .setInputCol("rawText")
      .setOutputCol("tokens")
    val dataset = sqlContext.createDataFrame(Seq(
      TokenizerTestData("Harold's not around.", Array("harold", "around")),
      TokenizerTestData("The dog's nose KNOWS!", Array("dog", "nose", "knows"))
    ))
    testLuceneAnalyzer(analyzer, dataset)
  }

  test("UAX29URLEmailTokenizer") {
    val analyzerConfig = """
                           |{
                           |  "schemaType": "LuceneAnalyzerSchema.v1",
                           |  "analyzers": [{
                           |    "name": "uax29urlemail_2000",
                           |    "tokenizer": {
                           |      "type": "uax29urlemail",
                           |      "maxTokenLength": "2000"
                           |    }
                           |  }],
                           |  "inputColumns": [{
                           |    "regex": ".+",
                           |    "analyzer": "uax29urlemail_2000"
                           |  }]
                           |}""".stripMargin
    val analyzer = new LuceneAnalyzer()
      .setAnalysisSchema(analyzerConfig)
      .setInputCol("rawText")
      .setOutputCol("tokens")
    val dataset = sqlContext.createDataFrame(Seq(
      TokenizerTestData("Click on https://www.google.com/#q=spark+lucene",
        Array("Click", "on", "https://www.google.com/#q=spark+lucene")),
      TokenizerTestData("Email caffeine@coffee.biz for tips on staying@alert",
        Array("Email", "caffeine@coffee.biz", "for", "tips", "on", "staying", "alert"))
    ))
    testLuceneAnalyzer(analyzer, dataset)
  }
}

object LuceneAnalyzerSuite extends SparkFunSuite {

  def testLuceneAnalyzer(t: LuceneAnalyzer, dataset: DataFrame): Unit = {
    logError(">>>>> Columns: " + dataset.columns.mkString(", "))
    t.transform(dataset)
      .select("tokens", "wantedTokens")
      .collect()
      .foreach { case Row(tokens, wantedTokens) =>
        assert(tokens === wantedTokens)
      }
  }
}
