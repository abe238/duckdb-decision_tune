#pragma once

#include "duckdb/common/types/value.hpp"

namespace duckdb {
namespace decision_tune {

//! The text the model reads for a state, as `state_text` of the reference engine renders it: text as is; a STRUCT,
//! a MAP or a JSON object as `key: value | key: value` (text values as is, NULL as empty, other values as compact
//! JSON); any other value as compact JSON (dates, timestamps and the like as their text).
string StateText(const Value &value);

//! One option of a decision: the key returned to the caller and the text the model reads.
struct DecisionOption {
	string key;
	string text;
};

//! The options of `decision_choose`: a LIST of texts (each its own key), or a MAP or STRUCT of key -> description
//! (an empty or NULL description reads as the key with `_` as spaces). Throws InvalidInputException unless there
//! are at least two distinct, non-empty keys.
vector<DecisionOption> DecisionOptions(const Value &options);

} // namespace decision_tune
} // namespace duckdb
