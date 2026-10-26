#pragma once

namespace duckdb {
class DatabaseInstance;

namespace decision_tune {

//! Registers the "decision_tune" grammar extension (`FROM t DECIDE (...)`,
//! `decision_choose(question AMONG (...) GIVEN ...)` and `decision_yes_no(question GIVEN ...)`) and activates it on
//! every connection.
void RegisterDecideGrammar(DatabaseInstance &db);

} // namespace decision_tune
} // namespace duckdb
