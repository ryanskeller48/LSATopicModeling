
import scala.math
import breeze.linalg._

class Term_x_Document (var input: Array[String]) { 
/** input is an array of all the documents we wish to include in a Term x Document matrix (plus do other analysis on later)
* 
*/
  val inputsize = input.length
  var wordcounts:Array[Int] = new Array[Int](inputsize) // holds the wordcount for each document
  var z = 0
  while (z < inputsize) {
    wordcounts(z) = input(z).split(" ").length
    z = z + 1    
  }
  var masterwordlist:Array[String] = new Array(0) // need an ordered list of the words as they are the index of the rows of the tf-idf matrix
 
  def term_x_newdoc (doc: String): Map[String, Int] = { 
  /**
  *  returns the Term x Document for the first document being compared
  *  the next function will use the keyset built here
  */
    var terms:Map[String, Int] = Map()
    // output map containing terms and their frequency (we will add to this as the document is parsed)
    var words:Array[String] = doc.split(" ") /**
    *  text is assumed to have no punctuation, containing only words separated by spaces
    *  (the text will be pre-processed elsewhere to achieve this)
    *  this splits the text into an array containing the individual words of the text
    */
    var numwords:Int = words.length // number of words in the document
    var i:Int = 0
    var word:String = "" // holds word we are currently processing
    var count:Int = 0
    while (i < numwords) {
      word = words(i) // get the i-th word in the array
      //println(word);
      if (terms.contains(word)) { // if the word is already in the Term x Document matrix...
        count = terms(word)
        terms = terms + (word -> (count+1)) // ...increment the count of the frequency of that word
      }
      else { 
        terms = terms + (word -> 1) // if it is not in the Term x Document matrix, add it with a count of 1
        //println(terms("the"));
      }
      i = i + 1
    }
    return terms
  }
  
  def term_x_doc (doc: String, prev: Set[String]): Map[String, Int] = {  // used to continue adding to the Term x Document database once the first one is made
    var terms:Map[String, Int] = Map() 
    prev.foreach((s: String) => (terms += (s -> 0))) /**
    *  we start this Term x Document matrix by importing the words from the previous Term x Document
    *  when we reach the end, the last document's Term x Document matrix will contain all the words from all documents,
    *  which is useful when formatting output/comparing each document
    */
    var words:Array[String] = doc.split(" ")
    var numwords:Int = words.length
    var i:Int = 0
    var word:String = ""
    var count:Int = 0

    while (i < numwords) {
      word = words(i)
      if (terms.contains(word)) {
        count = terms(word)
        terms = terms + (word -> (count+1))
      }
      else { 
        terms = terms + (word -> 1)
      }
      i = i + 1
    }
    return terms    
  }
  
  def analyze: Array[Map[String, Int]] = {
    var len:Int = input.length // number of documents to analyze
    if (len == 0) { return null } // if there are no documents there is no analysis
    var out:Array[Map[String, Int]] = new Array[Map[String, Int]](len) // output array containing the Term x Document matrices for each corresponding input document
    var i:Int = 1
    var terms:Map[String, Int] = term_x_newdoc(input(0)) // the Term x Document matrix for the first document, which will be used to seed the others
    out(0) = terms
    var keylist:Set[String] = terms.keySet // the unique words from the first document used to seed the next Term x Document
    while (i < len) {
      terms = term_x_doc(input(i), keylist) // get the Term x Document for the i-th document, seeded by all words that have appeared before
      keylist = terms.keySet // continue to update word list
      out(i) = terms // the i-th document gets its Term x Document assigned to the i-th slot of output
      i = i + 1
    }
    return out
  }
  
  def outMatrix(terms:Array[Map[String, Int]]): DenseMatrix[Int] = {
     /***
     * Converts array of maps to a Dense Matrix with the documents on the x axis and terms on the y axis.
     * Values in the matrix are frequency counts.
     */

    var len:Int = terms.length // number of documents
    var word:String = "" // current word being worked on
    var i:Int = 0 // index to access each document
    var j:Int = 0 // index to access each term
    val terms_list = terms(len-1).keys.toList // take terms out of map 
    val values_list = terms(len-1).values.toList // take values out of map
    var map:Map[String, Int] = Map() // variable to hold the map for each array element
    val freq_matrix = DenseMatrix.zeros[Int](terms_list.size,len) // create matrix of proper dimensions

    while (j < terms_list.size) { // for all words on the master list
      i = 0
      word = terms_list(j)
      while (i < len) { // for all documents
        map = terms(i) // access the Term x Document for the i-th document
        if (map.contains(word)) {
          freq_matrix(j,i) = map(word) 
        } // don't need else because matrix is initialized to zero 
        i = i + 1 // get the next document
      }
      j += 1 // get the next term
    }
    return freq_matrix
  }

  // calculates inverse term frequencymatrix, weighting uncommon words more heavily than common words 
  def tf_idf(term_x_doc_freq: DenseMatrix[Int], terms: Array[Map[String,Int]]): DenseMatrix[Double] = {
    var len:Int = terms.length // number of documents
    var num_rows:Int = term_x_doc_freq.rows 
    var num_cols:Int = term_x_doc_freq.cols
    var num_docs = DenseVector.zeros[Double](num_rows) // holds the number of docs every term appears in
    var num_words = DenseVector.zeros[Double](num_cols) // holds the number of words in each doc 
    //var keylist:Set[String] = terms(terms.length-1).keySet // master list of keys
    var i:Int = 0 // iterator
    var j:Int = 0 // iterator
    var count:Int = 0 // tmp value
    var doc_density:Double = 0 // tmp value to help with weighting calculations
    var new_matrix = DenseMatrix.zeros[Double](num_rows,num_cols) // matrix to hold tf-idf values initialized to 0

    val terms_list = terms(len-1).keys.toList // get keys 
    val values_list = terms(len-1).values.toList // get values for keys

    /* for each term, count the number of documents in which it appears in num_docs. For each doc, count the number of words in
     * num_words.
     */

    // takes care of both how many documents each term appears in and how many words in each document at once
    while (i < num_rows) {
      count = 0
      while (j < num_cols) {
        if (term_x_doc_freq(i,j) > 0) { // if value is positive, it appears in this document
          count += 1 // increase count
        }
        num_words(j) += term_x_doc_freq(i,j) // increase number of words in document j
        j += 1
      }
      //if(count == 0) {print("OHNO")} // no word should appear in zero documents
      num_docs(i) = count // add number of documents term appears in to array
      j = 0
      i += 1
    }

    i = 0
    import scala.math

    // calculate the tf-idf value for each term x document cell
    while (i < num_rows) {
      j = 0
      while (j < num_cols) {
        doc_density = (num_cols / num_docs(i)) // calculate doc density: total number of documents / number of document given term appears in
        new_matrix(i,j) = (term_x_doc_freq(i,j) / num_words(j)) * scala.math.log(doc_density)
        //new_matrix(i,j) = (scala.math.log(1 + term_x_doc_freq(i,j))) * scala.math.log(doc_density) // insert tf-idf value
        j += 1
      }
      i += 1
    }
    return new_matrix
  }

  def fastTruncatedSVD(A: DenseMatrix[Double], k: Int) = {
    val Omega = randn(A.cols*k).toDenseMatrix.reshape(A.cols,k)
    val Q = qr(A * Omega).q(::, 0 until k)
    val svdB = svd(Q.t * A)
    val U = Q * svdB.U
    (U,svdB.S,svdB.Vt)
  }

  def SVD_wrapper(tfidf: DenseMatrix[Double], k:Int) = {
    /*
     * tfidf is the dense matrix representing the inverse frequency matrix. k is the number of files the program is working with.
     */
    var dimensions:Int = 2
    var i:Int = 2
    var j:Int = 0
    var checked_all = 0
    var drop_found = 0
    var t = fastTruncatedSVD(tfidf, k)
    //var term_x_topic:DenseMatrix = new DenseMatrix[Double]
    //var topic_x_doc:DenseMatrix = new DenseMatrix[Double]
    //var single_vals:DenseVector = new DenseVector[Double]

    while ((dimensions <= k) && (checked_all == 0) && (drop_found == 0)) {
      if (dimensions == k) {checked_all = 1} // if dimensions = k, we have checked everything and want this to be last iteration
      t = fastTruncatedSVD(tfidf,dimensions)
      i = 1
      while ((i < dimensions) && (drop_found == 0)) {
        // in the following lines, index i of single_vals correspinds to dimension i + 1
        if ((t._2(i) / t._2(i - 1)) < .05) { // if there is a steep drop between dimension i and i + 1
          drop_found = 1
          t = fastTruncatedSVD(tfidf, i) // re-do SVD with dimension i 
        }
        // if a significant drop has been found, optimize if necessary and return with appropriate dimensions
        if (drop_found == 1) {
          j = i
          // if error is too large, re-do SVD by increments of 1 until error is either acceptable or we're back at max dimensions
          while ((norm((tfidf - (t._1 * diag(t._2) * t._3)).toDenseVector) > .001) && (j < k)) {
            j += 1
            t = fastTruncatedSVD(tfidf, j)
          }
          //(term_x_topic, topic_x_doc) // return values
        }
        i += 1
      }
      if (dimensions + 5 > k) {
        dimensions = k
      } else {dimensions += 5}
    }
    // if we're back to max dimensions after no significant drop, return with max dimensions
    (t._1, t._3)
  }

  def analyze_SVD(term_x_topic: DenseMatrix[Double], topic_x_doc: DenseMatrix[Double], terms: Array[Map[String,Int]], files: Array[String]) = {
    var len:Int = terms.length // number of documents
    val terms_list = terms(len-1).keys.toList // get master list of terms
    var i:Int = 0 // iterator
    var j:Int = 0 // iterator
    var num_topics = term_x_topic.cols // number of "latent topics" as per the dimensions of the SVD decomposition
    var terms_in_topics:Array[Map[String, Double]] = new Array[Map[String, Double]](num_topics) // for each topic, map contributing terms to their weights
    var topics_in_docs:Array[Map[Int, Double]] = new Array[Map[Int, Double]](num_topics) // for each topic, map contributing documents to their weights
    var tmp_terms_map:Map[String,Double] = Map() // tmp map for values in the terms_in_topics array
    var tmp_docs_map:Map[Int,Double] = Map() // tmp map for values in the topics_in_docs array
    var top_terms:Int = 0 // cutoff point for terms
    var top_docs:Int = 0 // cutoff point for docs

    // for every topic, find all contributing words and put them in the term->weight map for that topic
    while (i < term_x_topic.cols) {
      j = 0
      while (j < term_x_topic.rows) {
        if (term_x_topic(j,i) > 0) {
          tmp_terms_map +=  (terms_list(j) -> term_x_topic(j,i))
        }
        j += 1
      }
      terms_in_topics(i) = tmp_terms_map // insert into array
      tmp_terms_map = tmp_terms_map.empty // clear tmp
      i += 1
    } 

    i = 0
    // for every topic, find all documents that it exists in and put them in the document->weight map for that topic
    while (i < topic_x_doc.rows) {
      j = 0
      while (j < topic_x_doc.cols) {
        //if(i == 0) {print("YES1")}
        if (topic_x_doc(i,j) > 0) {
          //if(i == 0) {print("YES2")}
          tmp_docs_map += (j -> topic_x_doc(i,j))
        }
        j += 1
      }
      topics_in_docs(i) = tmp_docs_map // insert into array
      //if (i == 0) {print("DOCS" + topics_in_docs(i))}
      tmp_docs_map = tmp_docs_map.empty // clear map
      i += 1
    } 

    i = 0
    // for every topic, print most signficant contributing terms and documents in which topics are most prominent
    while (i < num_topics) {
      var tmp_terms = terms_in_topics(i).toSeq.sortBy(-_._2) // sort terms by weight
      var tmp_docs = topics_in_docs(i).toSeq.sortBy(-_._2) // sort topics by weight
      top_terms = math.ceil(tmp_terms.size * .05).toInt // find cutoff point for terms, top 5%
      top_docs = math.ceil(tmp_docs.size * .5).toInt // find cutoff point for documents, top 50%

      print("\nTopic #" + (i + 1) + " includes terms (most significant 5%): \n[");
      j = 0

      while (j < top_terms) {
        if ((j != 0) && (j != top_terms)) {print(", ")}
        print(tmp_terms(j)._1);
        j += 1
      }

      print("]\n\nTopic #" + (i + 1) + " appears in documents (most significant 50%): \n[");
      j = 0
      while (j < top_docs) {
        if ((j != 0) && (j != top_docs)) {print(", ")}
        print(files(tmp_docs(j)._1));
        j += 1
      }
      i += 1
      println("]\n\n===========================================================");
    }
  }

}
  object Term_x_Document {
    import java.io.File
    import scala.io.Source
    import java.nio.charset.CodingErrorAction
    import scala.io.Codec 

    // handle strange characters that made it into the text files
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    def main(args: Array[String]): Unit = {
      val files = getListOfFiles("src/main/resources") // get list of files
      val num_files:Int = files.length
      var i:Int = 0
      var j:Int = 0 // for this weird .DS Store file 
      var num_real_files:Int = num_files

      // mac OS sometimes puts strange .DS_Store files into folders, so filter it out if there
      while(j < num_files) {
        if((files(j).getPath() == "src/main/test_resources/.DS_Store") || 
          (files(j).getPath() == "src/main/resources/.DS_Store")) {
          num_real_files -= 1
        }
        j += 1
      }

      j = 0

      val file_text:Array[String] = new Array[String](num_real_files)
      val file_names:Array[String] = new Array[String](num_real_files)

      // for each file in the resources folder, save text as a string in an array
      while (j < num_files) {
        if((files(j).getPath() != "src/main/test_resources/.DS_Store") && 
          (files(j).getPath() != "src/main/resources/.DS_Store")) {
          file_text(i) = Source.fromFile(files(j).getPath()).mkString
          file_names(i) = files(j).getPath().mkString.replace("src/main/resources/","")
          i += 1
        }
        j += 1
      }
      
      val txd = new Term_x_Document(file_text)
      val out = txd.analyze
      val str = txd.outMatrix(out)
      val new_m = txd.tf_idf(str,out)
      var (term_x_topic, topic_x_doc) = txd.SVD_wrapper(new_m,num_real_files)
      txd.analyze_SVD(term_x_topic, topic_x_doc, out, file_names)
     
    }

    // short helper function to get list of files 
    def getListOfFiles(dir: String):List[File] = {
      val d = new File(dir)
      if (d.exists && d.isDirectory) {
        d.listFiles.filter(_.isFile).toList
      } else {
        List[File]()
      }
    }
  }
 
