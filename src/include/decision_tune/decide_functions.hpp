#pragma once

namespace duckdb {
class ExtensionLoader;

namespace decision_tune {

//! Registers `decision_choose(state, question, options)`, `decision_yes_no(state, question)` and
//! `decision_tune_tokenize(text)`.
void RegisterDecideFunctions(ExtensionLoader &loader);

} // namespace decision_tune
} // namespace duckdb
