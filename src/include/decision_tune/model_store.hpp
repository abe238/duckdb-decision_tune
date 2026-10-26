#pragma once

#include "decision_tune/decision_model.hpp"

#include "duckdb.hpp"

namespace duckdb {
class ExtensionLoader;

namespace decision_tune {

struct ModelFile;

//! Resolves, downloads, verifies and caches the DecisionTune files, and keeps one tokenizer and one ONNX Runtime
//! session per model directory for the whole process.
class ModelStore {
public:
	static ModelStore &Instance();

	//! The tokenizer alone (only `tokenizer.json` is needed).
	shared_ptr<BpeTokenizer> GetTokenizer(ClientContext &context);
	//! The model: `decision_tune_model_path` if set, otherwise the cached files (downloaded into
	//! `decision_tune_cache_dir` from `decision_tune_model_url` if missing). Cached and downloaded files are checked
	//! against the SHA-256 in the release manifest.
	shared_ptr<DecisionModel> GetModel(ClientContext &context);

	//! `<cache_dir>/<file_name>` with `~` expanded.
	static string CachePath(ClientContext &context, const ModelFile &file);
	static string FileUrl(ClientContext &context, const ModelFile &file);
	//! True if the cache holds a file with the final name and the manifest size (contents not checked).
	static bool IsCached(ClientContext &context, const ModelFile &file);

private:
	struct ModelRecord {
		mutex lock;
		bool tokenizer_ready = false;
		bool onnx_ready = false;
		shared_ptr<BpeTokenizer> tokenizer;
		shared_ptr<DecisionModel> model;
	};
	struct Location {
		//! The directory the files are read from: the cache or `decision_tune_model_path`.
		string directory;
		bool explicit_path = false;
	};

	Location Resolve(ClientContext &context);
	shared_ptr<ModelRecord> Record(const string &directory);
	//! The local path of `file`, downloaded and verified first when it comes from the cache. Caller holds the
	//! record lock.
	string EnsureFile(ClientContext &context, const Location &location, const ModelFile &file, bool &ready);
	shared_ptr<BpeTokenizer> LoadTokenizer(ClientContext &context, const Location &location, ModelRecord &record);

	//! Only protects record lookup; file I/O and session construction use the record's lock.
	mutex lock;
	unordered_map<string, shared_ptr<ModelRecord>> models;
};

//! Registers the `decision_tune_*` settings and the `decision_tune_models()` table function.
void RegisterModelStore(ExtensionLoader &loader);

} // namespace decision_tune
} // namespace duckdb
