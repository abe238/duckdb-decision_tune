#pragma once

#include "decision_tune/scorer.hpp"
#include "decision_tune/tokenizer.hpp"

#include "duckdb/common/list.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"

#include <utility>
#include <vector>

namespace duckdb {
namespace decision_tune {

//! The model input for one question: `[CLS] question [SEP] ([MASK] option)+ [SEP] state [SEP]`.
struct Prompt {
	std::vector<int64_t> ids;
	//! The position of the `[MASK]` before each option.
	std::vector<int64_t> positions;
};

//! `DecisionModel.encode` of the reference engine. Throws InvalidInputException when the prompt exceeds the
//! context (the reference refuses it too, there is no truncation).
Prompt EncodePrompt(const BpeTokenizer &tokenizer, const string &state, const string &question,
                    const vector<string> &options);
//! The ids of `text` as it appears in a prompt: the reference replaces the special tokens `[MASK]`, `[CLS]`,
//! `[SEP]`, `[PAD]` and `[UNK]` with spaces first.
std::vector<int64_t> EncodeText(const BpeTokenizer &tokenizer, const string &text);

//! DecisionTune 1.0: the tokenizer, the network and a memo of recent answers. Thread-safe.
class DecisionModel {
public:
	DecisionModel(shared_ptr<BpeTokenizer> tokenizer, shared_ptr<Scorer> scorer);

	const BpeTokenizer &Tokenizer() const {
		return *tokenizer;
	}
	//! The softmax over the option logits of `prompt`. Recent prompts are answered from memory, so a query that
	//! evaluates the same decision twice (or runs again) does not pay for a second forward pass.
	std::vector<double> Probabilities(const Prompt &prompt);

private:
	using CacheKey = std::pair<uint64_t, uint64_t>;
	struct CacheKeyHash {
		size_t operator()(const CacheKey &key) const {
			return static_cast<size_t>(key.first ^ (key.second * 0x9E3779B97F4A7C15ULL));
		}
	};
	using CacheList = list<std::pair<CacheKey, std::vector<double>>>;

	bool Lookup(const CacheKey &key, std::vector<double> &result);
	void Store(const CacheKey &key, const std::vector<double> &result);

	shared_ptr<BpeTokenizer> tokenizer;
	shared_ptr<Scorer> scorer;

	mutex cache_lock;
	//! Most recently used first.
	CacheList cache_entries;
	unordered_map<CacheKey, CacheList::iterator, CacheKeyHash> cache_index;
};

} // namespace decision_tune
} // namespace duckdb
