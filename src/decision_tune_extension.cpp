#define DUCKDB_EXTENSION_MAIN

#include "decision_tune_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "decision_tune/decide_functions.hpp"
#include "decision_tune/decide_grammar.hpp"
#include "decision_tune/model_manifest.hpp"
#include "decision_tune/model_store.hpp"
#include "decision_tune/scorer.hpp"

namespace duckdb {

static void DecisionTuneVersionFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &children = StructVector::GetEntries(result);
	DecisionTuneExtension extension;
	auto extension_version = extension.Version();
	children[0].SetValue(0, Value(extension_version.empty() ? "dev" : extension_version));
	children[1].SetValue(0, Value(decision_tune::MODEL_NAME));
	children[2].SetValue(0, Value(decision_tune::OrtScorer::RuntimeVersion()));
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
}

static void LoadInternal(ExtensionLoader &loader) {
	auto version_type = LogicalType::STRUCT(
	    {{"extension", LogicalType::VARCHAR}, {"model", LogicalType::VARCHAR}, {"onnxruntime", LogicalType::VARCHAR}});
	ScalarFunction version("decision_tune_version", {}, version_type, DecisionTuneVersionFun);
	version.SetStability(FunctionStability::CONSISTENT);
	loader.RegisterFunction(version);

	decision_tune::RegisterModelStore(loader);
	decision_tune::RegisterDecideFunctions(loader);
	decision_tune::RegisterDecideGrammar(loader.GetDatabaseInstance());
}

void DecisionTuneExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DecisionTuneExtension::Name() {
	return "decision_tune";
}

std::string DecisionTuneExtension::Version() const {
#ifdef EXT_VERSION_DECISION_TUNE
	return EXT_VERSION_DECISION_TUNE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(decision_tune, loader) {
	duckdb::LoadInternal(loader);
}
}
