#include "decision_tune/decision_model.hpp"

#include "decision_tune/model_manifest.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/hash.hpp"

#include <cmath>

namespace duckdb {
namespace decision_tune {

//! Memoized answers; a few MB for typical prompts.
static constexpr idx_t CACHE_ENTRIES = 100000;

std::vector<int64_t> EncodeText(const BpeTokenizer &tokenizer, const string &text) {
	auto cleaned = text;
	for (auto special : {"[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]"}) {
		cleaned = StringUtil::Replace(cleaned, special, " ");
	}
	return tokenizer.Encode(cleaned);
}

static void Append(std::vector<int64_t> &target, const std::vector<int64_t> &source) {
	target.insert(target.end(), source.begin(), source.end());
}

Prompt EncodePrompt(const BpeTokenizer &tokenizer, const string &state, const string &question,
                    const vector<string> &options) {
	auto cls = tokenizer.TokenId("[CLS]");
	auto sep = tokenizer.TokenId("[SEP]");
	auto mask = tokenizer.TokenId("[MASK]");
	Prompt prompt;
	prompt.ids.push_back(cls);
	Append(prompt.ids, EncodeText(tokenizer, question));
	prompt.ids.push_back(sep);
	for (auto &option : options) {
		prompt.positions.push_back(static_cast<int64_t>(prompt.ids.size()));
		prompt.ids.push_back(mask);
		Append(prompt.ids, EncodeText(tokenizer, option));
	}
	prompt.ids.push_back(sep);
	Append(prompt.ids, EncodeText(tokenizer, state));
	prompt.ids.push_back(sep);
	if (prompt.ids.size() > CONTEXT_TOKENS) {
		throw InvalidInputException("decision_tune: the prompt is %llu tokens, more than the %llu-token context of %s. "
		                            "Shorten the input, e.g. left(message, 20000)",
		                            prompt.ids.size(), CONTEXT_TOKENS, MODEL_NAME);
	}
	return prompt;
}

DecisionModel::DecisionModel(shared_ptr<BpeTokenizer> tokenizer_p, shared_ptr<Scorer> scorer_p)
    : tokenizer(std::move(tokenizer_p)), scorer(std::move(scorer_p)) {
}

static uint64_t Fnv1a(const_data_ptr_t data, idx_t size) {
	uint64_t hash = 0xcbf29ce484222325ULL;
	for (idx_t i = 0; i < size; i++) {
		hash = (hash ^ data[i]) * 0x100000001b3ULL;
	}
	return hash;
}

std::vector<double> DecisionModel::Probabilities(const Prompt &prompt) {
	// Two independent 64-bit hashes of the whole input stand in for the input itself.
	std::vector<int64_t> key_data(prompt.ids);
	key_data.push_back(-1);
	Append(key_data, prompt.positions);
	auto bytes = reinterpret_cast<const_data_ptr_t>(key_data.data());
	auto size = key_data.size() * sizeof(int64_t);
	CacheKey key {Hash(reinterpret_cast<const char *>(bytes), size), Fnv1a(bytes, size)};

	std::vector<double> result;
	if (Lookup(key, result)) {
		return result;
	}
	auto logits = scorer->Logits(prompt.ids, prompt.positions);
	// Softmax in double precision, as the reference does with Python floats.
	double top = -INFINITY;
	for (auto logit : logits) {
		top = MaxValue<double>(top, logit);
	}
	double total = 0;
	result.reserve(logits.size());
	for (auto logit : logits) {
		result.push_back(std::exp(static_cast<double>(logit) - top));
		total += result.back();
	}
	for (auto &p : result) {
		p /= total;
	}
	Store(key, result);
	return result;
}

bool DecisionModel::Lookup(const CacheKey &key, std::vector<double> &result) {
	lock_guard<mutex> guard(cache_lock);
	auto entry = cache_index.find(key);
	if (entry == cache_index.end()) {
		return false;
	}
	cache_entries.splice(cache_entries.begin(), cache_entries, entry->second);
	result = entry->second->second;
	return true;
}

void DecisionModel::Store(const CacheKey &key, const std::vector<double> &result) {
	lock_guard<mutex> guard(cache_lock);
	if (cache_index.find(key) != cache_index.end()) {
		return;
	}
	cache_entries.emplace_front(key, result);
	cache_index[key] = cache_entries.begin();
	if (cache_entries.size() > CACHE_ENTRIES) {
		cache_index.erase(cache_entries.back().first);
		cache_entries.pop_back();
	}
}

} // namespace decision_tune
} // namespace duckdb
