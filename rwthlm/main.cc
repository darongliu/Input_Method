/*
 * Copyright 2014 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH LM License (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include "data.h"
#include "gradienttest.h"
#include "htklatticerescorer.h"
#include "trainer.h"
#include "vocabulary.h"

namespace po = boost::program_options;

void ParseCommandLine(const int argc,
                      const char *const argv[],
                      po::variables_map *options) {
  // define command line options
  po::options_description visible("Command line options"), hidden, all;
  visible.add_options()
      ("help", "produce this help message")
      ("config", po::value<std::string>()->default_value(""), "config file")
      ("verbose", "verbose program output")
      ("vocab", po::value<std::string>(), "vocabulary file")
      ("remap", po::value<std::string>(), "remapped vocabulary file")
      ("unk", "use closed vocabulary")
      ("map-unk", po::value<std::string>()->default_value("<unk>"),
       "name of unknown token")
      ("map-sb", po::value<std::string>()->default_value("<sb>"),
       "name of sentence boundary token")
      ("debug-no-sb", "do not insert <sb> tokens automatically (debug only!)")
      ("train", po::value<std::string>(), "training data file")
      ("dev", po::value<std::string>(), "development data file")
      ("ppl", po::value<std::string>(), "data file for computing perplexity")
      ("random-seed", po::value<uint32_t>()->default_value(1),
       "random number generator seed")
      ("learning-rate", po::value<Real>(), "initial learning rate")
      ("momentum", po::value<Real>()->default_value(0.0),
       "momentum parameter")
      ("batch-size", po::value<int>()->default_value(1),
       "maximum number of sequences evaluated in parallel")
      ("sequence-length", po::value<int>()->default_value(100),
       "maximum length of a sequence")
      ("max-epoch", po::value<int>()->default_value(0),
       "maximum number of epochs to train, zero means unlimited")
      ("no-shuffling", "do not shuffle training data")
      ("word-wrapping",
       po::value<std::string>()->default_value("fixed"),
       "concatenated, fixed or verbatim")
      ("feedforward", "training in feedforward style without recurrencies")
      ("no-bias", "do not use any bias")
      ("num-oovs", po::value<size_t>()->default_value(0),
       "difference in recognition and neural network LM vocabulary size")
      ("lambda", po::value<Real>(),
       "interpolation weight of neural network LM")
      ("look-ahead-semiring",
       po::value<std::string>()->default_value("none"),
       "none, tropical or log semiring")
      ("dependent", "use previous best state for rescoring current lattice")
      ("look-ahead-lm-scale", po::value<Real>(),
       "Look ahead LM scale for lattice decoding (default: lm-scale)")
      ("lm-scale", po::value<Real>()->default_value(1.0),
       "LM scale for lattice decoding")
      ("pruning-threshold", po::value<Real>(),
       "beam pruning threshold for lattice rescoring, zero means unlimited")
      ("pruning-limit", po::value<size_t>()->default_value(0),
       "maximum number of hypotheses per lattice node, zero means unlimited")
      ("dp-order", po::value<int>()->default_value(3),
       "dynamic programming order for lattice rescoring")
      ("output", po::value<std::string>()->default_value("lattice"),
       "ctm, lattice, or expanded-lattice")
      ("clear-initial-links",  // some options for compatibility with RWTH ASR software
       "set scores of initial links in a lattice to zero")
      ("set-sb-next-to-last",
       "set link label of next to last links in a lattice to <sb>")
      ("set-sb-last",
       "set link label of last links in a lattice to <sb>");

  hidden.add_options()
      ("positional", po::value<std::vector<std::string>>(),
          "positional arguments")
      ("self-test", "compare gradient to difference quotient");
  all.add(visible).add(hidden);

  // define positional options
  po::positional_options_description positional_description;
  positional_description.add("positional", -1);

  try {
    // parse command line options
    po::store(po::command_line_parser(argc, argv).
	    options(all).
		positional(positional_description).
        run(), *options);
    po::notify(*options);

    // config file?
    const std::string config_file = (*options)["config"].as<std::string>();
    if (config_file != "") {
      assert(boost::filesystem::exists(config_file));
      std::ifstream file(config_file, std::ifstream::in);
      assert(file.good());
      std::cout << "Reading options from config file '" << config_file <<
                   "' ..." << std::endl;
      po::store(po::parse_config_file(file, all), *options);
    }

    // help option?
	if (options->count("help") || !options->count("positional")) {
      std::cout << "Usage: rwthlm [OPTION]... [LATTICE]... NETWORK\n";
      std::cout << visible;
      exit(0);
    }
  } catch (std::exception &e) {
    // unable to parse: print error message
    std::cerr << e.what() << '\n';
    exit(1);
  }
}

void EvaluateCommandLine(const po::variables_map &options) {
  try {
    // parse positional arguments
    std::vector<std::string> positional = options["positional"].
        as<std::vector<std::string>>();
    assert(positional.size() >= 1);
    std::string net_config = positional.back();
    positional.pop_back();

    // parse arguments for which default values have been defined
    const bool is_feedforward = options.count("feedforward") > 0,
               debug_no_sb = options.count("debug-no-sb") > 0;
    const uint32_t seed = options["random-seed"].as<uint32_t>();
    assert(seed >= 0);
    const size_t num_oovs = options["num-oovs"].as<size_t>();
    const int max_batch_size = options["batch-size"].as<int>(),
              max_sequence_length = options["sequence-length"].as<int>(),
              max_epoch = options["max-epoch"].as<int>();
    const Real momentum = options["momentum"].as<Real>();
    const std::string unk = options.count("unk") ? 
                            options["map-unk"].as<std::string>() : "",
                      sb = options["map-sb"].as<std::string>();
    WordWrappingType word_wrapping_type;
    const std::string type = options["word-wrapping"].as<std::string>();
    if (type == "concatenated")
      word_wrapping_type = kConcatenated;
    else if (type == "fixed")
      word_wrapping_type = kFixed;
    else if (type == "verbatim")
      word_wrapping_type = kVerbatim;
    else
      assert(false);

    // parse data file names (for training, development, test/perplexity)
    std::string train_file;
    if (options.count("train"))
      train_file = options["train"].as<std::string>();
    std::string dev_file;
    if (options.count("dev"))
      dev_file = options["dev"].as<std::string>();
    std::string ppl_file;
    if (options.count("ppl"))
      ppl_file = options["ppl"].as<std::string>();

    // set up vocabulary
    ConstVocabularyPointer vocabulary;
    if (options.count("vocab")) {
      const std::string vocab_file = options["vocab"].as<std::string>();
      if (boost::filesystem::exists(vocab_file)) {
        std::cout << "Reading vocabulary from file '" <<
            vocab_file << "' ..." << std::endl;
        vocabulary = Vocabulary::ConstructFromVocabFile(vocab_file, unk, sb);
      } else {
        assert(train_file != "");
        std::cout << "Creating vocabulary from training data file '" <<
                     train_file << "' ..." << std::endl;
        vocabulary = Vocabulary::ConstructFromTrainFile(train_file, unk, sb);
        std::cout << "Saving vocabulary to file '" << vocab_file << "' ..." <<
                     std::endl;
        vocabulary->Save(vocab_file);
      }
    } else {
      // set up vocabulary from scratch
      std::cout << "Creating vocabulary from training data file '" <<
          train_file << "' ..." << std::endl;
      vocabulary = Vocabulary::ConstructFromTrainFile(train_file, unk, sb);
    }
    if (options.count("remap")) {
      std::cout << "Writing remapped vocabulary ..." << std::endl;
      vocabulary->Save(options["remap"].as<std::string>());
    }

    // create neural network
    Random random(seed);
    const Real learning_rate = options.count("learning-rate") > 0 ?
                               options["learning-rate"].as<Real>() : 0.1;
    // a sequence length of 3 is enough for rescoring!
    NetPointer net(new Net(
        vocabulary,
        max_batch_size,
        positional.empty() && !is_feedforward ? max_sequence_length : 3,
        num_oovs,
        is_feedforward,
        learning_rate,
        momentum,
        &random));
    if (boost::filesystem::exists(net_config)) {
      net->BuildNetworkAndLoad(net_config,
                               options.count("no-bias") == 0);
    } else {
      // Rescoring: The neural network file must exist!
      assert(positional.empty());
      net->BuildNetworkAndRandomize(net_config,
                                    options.count("no-bias") == 0);
    }

    DataPointer dev_data;
    if (dev_file != "") {
      std::cout << "Reading development data from file '" << dev_file <<
                   "' ..." << std::endl;
      dev_data = std::make_shared<Data>(dev_file,
                                        max_batch_size,
                                        max_sequence_length,
                                        word_wrapping_type,
                                        debug_no_sb,
                                        vocabulary);
    }

    if (ppl_file != "") {
      std::cout << "Computing perplexity for file '" << ppl_file << "' ..." <<
                   std::endl;
      DataPointer ppl_data = std::make_shared<Data>(ppl_file,
                                                    max_batch_size,
                                                    max_sequence_length,
                                                    word_wrapping_type,
                                                    debug_no_sb,
                                                    vocabulary);
      std::cout << "perplexity:\n" << std::fixed << std::setprecision(20);
      // Perplexity evaluation: The neural network file must exist!
      assert(boost::filesystem::exists(net_config));
      Trainer trainer(max_epoch,
                      false,  // no shuffling here
                      options.count("verbose") > 0,
                      is_feedforward,
                      net_config,
                      net,
                      vocabulary,
                      ppl_data,
                      ppl_data,
                      &random);
      std::cout << trainer.ComputePerplexity(ppl_data) << '\n';
      exit(0);
    }

    // remaining positional arguments: lattices for rescoring
    assert(positional.empty() == !options.count("lambda") &&
           positional.empty() == !options.count("pruning-threshold"));
    if (!positional.empty()) {
      const Real lambda = options["lambda"].as<Real>();
      assert(lambda >= 0. && lambda <= 1.);

      HtkLatticeRescorer::LookAheadSemiring semiring;
      const std::string name = options["look-ahead-semiring"].as<std::string>();
      if (name == "tropical")
        semiring = HtkLatticeRescorer::kTropical;
      else if (name == "log")
        semiring = HtkLatticeRescorer::kLog;
      else if (name == "none")
        semiring = HtkLatticeRescorer::kNone;
      else
        assert(false);

      HtkLatticeRescorer::OutputFormat output_format;
      const std::string format = options["output"].as<std::string>();
      if (format == "ctm")
        output_format = HtkLatticeRescorer::kCtm;
      else if (format == "lattice")
        output_format = HtkLatticeRescorer::kLattice;
      else if (format == "expanded-lattice")
        output_format = HtkLatticeRescorer::kExpandedLattice;
      else
        assert(false);

      const Real lm_scale = options["lm-scale"].as<Real>(),
                 look_ahead_lm_scale = options.count("look-ahead-lm-scale") ==
                     0 ? lm_scale : options["look-ahead-lm-scale"].as<Real>(),
                 beam = options["pruning-threshold"].as<Real>();
      const size_t limit = options["pruning-limit"].as<size_t>();
      RescorerPointer rescorer(new HtkLatticeRescorer(
          vocabulary,
          net,
          output_format,
          num_oovs,
          lambda,
          semiring,
          look_ahead_lm_scale,
		  lm_scale,
          beam == 0. ? std::numeric_limits<Real>::infinity() : beam,
          limit == 0 ? std::numeric_limits<size_t>::max() : limit,
          options["dp-order"].as<int>(),
          options.count("dependent") != 0,
          options.count("clear-initial-links") != 0,
          options.count("set-sb-next-to-last") != 0,
          options.count("set-sb-last") != 0));
      rescorer->Rescore(positional);
      exit(0);
    }
    
    DataPointer train_data;
    if (train_file != "") {
      std::cout << "Reading training data from file '" << train_file <<
                   "' ..." << std::endl;
      train_data = std::make_shared<Data>(train_file,
                                          max_batch_size,
                                          max_sequence_length,
                                          word_wrapping_type,
                                          debug_no_sb,
                                          vocabulary);
      Trainer trainer(max_epoch,
                      options.count("no-shuffling") == 0,
                      options.count("verbose") > 0,
                      is_feedforward,
                      net_config,
                      net,
                      vocabulary,
                      train_data,
                      dev_data,
                      &random);
      if (options.count("self-test") > 0) {
        GradientTest test(seed, &trainer, is_feedforward);
        test.Test();
      } else {
        assert(dev_file != "");
        if (!boost::filesystem::exists(net_config) &&
            options.count("learning-rate") == 0) {
          trainer.AutoInitializeLearningRate(seed);
        }
        trainer.Train(seed);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    exit(1);
  }
}

int main(int argc, char* argv[]) {
  po::variables_map options;
  
  ParseCommandLine(argc, argv, &options);
  EvaluateCommandLine(options);

  return 0;
}
