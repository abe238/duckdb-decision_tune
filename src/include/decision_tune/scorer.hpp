#pragma once

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decision_tune {

//! One forward pass of the DecisionTune encoder plus its linear head.
class Scorer {
public:
	virtual ~Scorer() = default;
	//! The head's logit at each of `positions` (the `[MASK]` before every option) of the sequence `ids`.
	virtual std::vector<float> Logits(const std::vector<int64_t> &ids, const std::vector<int64_t> &positions) = 0;
};

//! Runs the published `onnx/model.onnx` with ONNX Runtime (CPU). `Logits` is thread-safe and single-threaded:
//! parallelism comes from DuckDB scoring many prompts at once.
class OrtScorer : public Scorer {
public:
	explicit OrtScorer(const string &onnx_path);
	~OrtScorer() override;

	std::vector<float> Logits(const std::vector<int64_t> &ids, const std::vector<int64_t> &positions) override;

	//! The linked ONNX Runtime version, e.g. "1.29.0".
	static string RuntimeVersion();

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace decision_tune
} // namespace duckdb
