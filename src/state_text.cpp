#include "decision_tune/state_text.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "yyjson.hpp"

#include <cmath>
#include <cstdlib>

namespace duckdb {
namespace decision_tune {

using namespace duckdb_yyjson; // NOLINT

//===--------------------------------------------------------------------===//
// Compact JSON, as json.dumps(value, ensure_ascii=False, separators=(",", ":"))
//===--------------------------------------------------------------------===//
static void AppendJsonString(string &out, const string &text) {
	out += '"';
	for (auto c : text) {
		auto byte = static_cast<unsigned char>(c);
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		case '\b':
			out += "\\b";
			break;
		case '\f':
			out += "\\f";
			break;
		default:
			if (byte < 0x20) {
				out += StringUtil::Format("\\u%04x", byte);
			} else {
				out += c;
			}
		}
	}
	out += '"';
}

static bool IsTextType(const LogicalType &type) {
	return type.id() == LogicalTypeId::VARCHAR || type.id() == LogicalTypeId::ENUM;
}

static bool IsNumericType(const LogicalType &type) {
	return type.IsNumeric() || type.id() == LogicalTypeId::BOOLEAN;
}

static void AppendJson(string &out, const Value &value);

static void AppendJsonNumber(string &out, const Value &value) {
	auto &type = value.type();
	if (type.id() == LogicalTypeId::BOOLEAN) {
		out += BooleanValue::Get(value) ? "true" : "false";
		return;
	}
	if (type.id() == LogicalTypeId::DOUBLE || type.id() == LogicalTypeId::FLOAT) {
		auto number = value.GetValue<double>();
		if (std::isnan(number)) {
			out += "NaN";
			return;
		}
		if (std::isinf(number)) {
			out += number > 0 ? "Infinity" : "-Infinity";
			return;
		}
	}
	out += value.ToString();
}

static void AppendJson(string &out, const Value &value) {
	if (value.IsNull()) {
		out += "null";
		return;
	}
	auto &type = value.type();
	if (IsNumericType(type)) {
		AppendJsonNumber(out, value);
		return;
	}
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY: {
		auto &children =
		    type.id() == LogicalTypeId::LIST ? ListValue::GetChildren(value) : ArrayValue::GetChildren(value);
		out += '[';
		for (idx_t i = 0; i < children.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			AppendJson(out, children[i]);
		}
		out += ']';
		return;
	}
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE: {
		auto &children = StructValue::GetChildren(value);
		if (StructType::IsUnnamed(type)) {
			out += '[';
			for (idx_t i = 0; i < children.size(); i++) {
				if (i > 0) {
					out += ',';
				}
				AppendJson(out, children[i]);
			}
			out += ']';
			return;
		}
		out += '{';
		for (idx_t i = 0; i < children.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			AppendJsonString(out, StructType::GetChildName(type, i).GetIdentifierName());
			out += ':';
			AppendJson(out, children[i]);
		}
		out += '}';
		return;
	}
	case LogicalTypeId::MAP: {
		out += '{';
		auto &entries = MapValue::GetChildren(value);
		for (idx_t i = 0; i < entries.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			auto &entry = StructValue::GetChildren(entries[i]);
			AppendJsonString(out, entry[0].ToString());
			out += ':';
			AppendJson(out, entry[1]);
		}
		out += '}';
		return;
	}
	default:
		AppendJsonString(out, value.ToString());
		return;
	}
}

static string CompactJson(const Value &value) {
	string out;
	AppendJson(out, value);
	return out;
}

//===--------------------------------------------------------------------===//
// state_text
//===--------------------------------------------------------------------===//
//! A value that is not a dict: text as is, NULL as empty, numbers, booleans and nested values as compact JSON, and
//! other scalars (dates, UUIDs, ...) as their text.
static string ScalarText(const Value &value) {
	if (value.IsNull()) {
		return string();
	}
	auto &type = value.type();
	if (IsTextType(type)) {
		return StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR));
	}
	if (IsNumericType(type) || type.IsNested()) {
		return CompactJson(value);
	}
	return value.ToString();
}

static string JoinFields(const vector<std::pair<string, string>> &fields) {
	string out;
	for (idx_t i = 0; i < fields.size(); i++) {
		if (i > 0) {
			out += " | ";
		}
		out += fields[i].first;
		out += ": ";
		out += fields[i].second;
	}
	return out;
}

//! A JSON document: an object reads like a STRUCT, a string as its text, anything else as compact JSON.
static string JsonStateText(const string &json) {
	auto doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
	if (!doc) {
		return json;
	}
	auto write = [](yyjson_val *val) {
		size_t length;
		auto text = yyjson_val_write(val, YYJSON_WRITE_NOFLAG, &length);
		string result = text ? string(text, length) : string();
		free(text);
		return result;
	};
	auto root = yyjson_doc_get_root(doc);
	string result;
	if (yyjson_is_obj(root)) {
		vector<std::pair<string, string>> fields;
		yyjson_obj_iter iter;
		yyjson_obj_iter_init(root, &iter);
		while (auto key = yyjson_obj_iter_next(&iter)) {
			auto val = yyjson_obj_iter_get_val(key);
			fields.emplace_back(string(yyjson_get_str(key), yyjson_get_len(key)),
			                    yyjson_is_str(val) ? string(yyjson_get_str(val), yyjson_get_len(val)) : write(val));
		}
		result = JoinFields(fields);
	} else if (yyjson_is_str(root)) {
		result = string(yyjson_get_str(root), yyjson_get_len(root));
	} else {
		result = write(root);
	}
	yyjson_doc_free(doc);
	return result;
}

string StateText(const Value &value) {
	if (value.IsNull()) {
		return string();
	}
	auto &type = value.type();
	if (type.IsJSONType()) {
		return JsonStateText(StringValue::Get(value));
	}
	if (type.id() == LogicalTypeId::STRUCT && !StructType::IsUnnamed(type)) {
		vector<std::pair<string, string>> fields;
		auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < children.size(); i++) {
			fields.emplace_back(StructType::GetChildName(type, i).GetIdentifierName(), ScalarText(children[i]));
		}
		return JoinFields(fields);
	}
	if (type.id() == LogicalTypeId::MAP) {
		vector<std::pair<string, string>> fields;
		for (auto &entry : MapValue::GetChildren(value)) {
			auto &key_value = StructValue::GetChildren(entry);
			fields.emplace_back(key_value[0].ToString(), ScalarText(key_value[1]));
		}
		return JoinFields(fields);
	}
	return ScalarText(value);
}

//===--------------------------------------------------------------------===//
// Options
//===--------------------------------------------------------------------===//
//! `option_text` of the reference for a {key: description} entry.
static string DescribedOptionText(const string &key, const Value &description) {
	if (!description.IsNull()) {
		auto text = description.ToString();
		if (!text.empty()) {
			return text;
		}
	}
	return StringUtil::Replace(key, "_", " ");
}

vector<DecisionOption> DecisionOptions(const Value &options) {
	vector<DecisionOption> result;
	auto &type = options.type();
	switch (type.id()) {
	case LogicalTypeId::LIST:
		for (auto &option : ListValue::GetChildren(options)) {
			if (option.IsNull()) {
				throw InvalidInputException("decision_tune: options must not contain NULL");
			}
			auto text = option.ToString();
			result.push_back({text, text});
		}
		break;
	case LogicalTypeId::MAP:
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &key_value = StructValue::GetChildren(entry);
			auto key = key_value[0].ToString();
			result.push_back({key, DescribedOptionText(key, key_value[1])});
		}
		break;
	case LogicalTypeId::STRUCT: {
		auto &children = StructValue::GetChildren(options);
		for (idx_t i = 0; i < children.size(); i++) {
			auto key = StructType::GetChildName(type, i).GetIdentifierName();
			result.push_back({key, DescribedOptionText(key, children[i])});
		}
		break;
	}
	default:
		throw InternalException("decision_tune: unexpected options type %s", type.ToString());
	}
	if (result.size() < 2) {
		throw InvalidInputException("decision_tune: a decision needs at least 2 options, got %llu", result.size());
	}
	unordered_set<string> keys;
	for (auto &option : result) {
		if (option.key.empty()) {
			throw InvalidInputException("decision_tune: options must not be empty");
		}
		if (!keys.insert(option.key).second) {
			throw InvalidInputException("decision_tune: option '%s' is given more than once", option.key);
		}
	}
	return result;
}

} // namespace decision_tune
} // namespace duckdb
