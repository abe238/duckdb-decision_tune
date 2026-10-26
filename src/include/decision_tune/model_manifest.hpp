#pragma once

#include <cstdint>

namespace duckdb {
namespace decision_tune {

//! A file of the DecisionTune 1.0 release, as listed in its `manifest.json`
//! (https://huggingface.co/decision-tune/decisiontune-1.0, Apache-2.0).
struct ModelFile {
	//! Name in the local cache directory.
	const char *file_name;
	//! Path in the Hugging Face repository.
	const char *remote_path;
	const char *sha256;
	uint64_t file_bytes;
};

static constexpr const char *MODEL_NAME = "DecisionTune 1.0";
static constexpr const char *MODEL_VERSION = "1.0";
static constexpr const char *HF_REPO = "decision-tune/decisiontune-1.0";
//! The repository commit the files below come from.
static constexpr const char *HF_REVISION = "27056cb08bcf2b3ba1fd9bccb782b4dfd0498e1f";

static constexpr ModelFile TOKENIZER_FILE = {
    "tokenizer.json", "tokenizer.json", "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30", 3583228ULL};
static constexpr ModelFile ONNX_FILE = {
    "model.onnx", "onnx/model.onnx", "cb78a8217d627fd7b17335c5d849184bd470d9c3d4dbc0537de7831162bd4c94", 1579877218ULL};
static constexpr const ModelFile *MODEL_FILES[] = {&TOKENIZER_FILE, &ONNX_FILE};

//! `max_position_embeddings` in config.json: longer prompts are refused, never truncated.
static constexpr uint64_t CONTEXT_TOKENS = 8192;

} // namespace decision_tune
} // namespace duckdb
