/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifndef MLIP_UTILS_H
#define MLIP_UTILS_H

#include <vector>
#include <memory.h>
#include <fstream>
#include <string>
#include <map>


// Minimum and maximum macros for compatibility with linux
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#define __min(a,b) (((a) < (b)) ? (a) : (b))

const int HUGE_INT = 999999999;
const double HUGE_DOUBLE = 9.9e99;

#ifndef _MLP_NOLOG
 #define MLP_LOG   ::LogOutput    // logging instruction
#else
 #define MLP_LOG(...) ((void)0)   // strip out logging instructions from code
#endif

void LogOutput(const char* szTag, const char* szFmt, ... );
void LogOutput(const char* szTag, const std::string& str );
int ParseCommonOptions(int argc, char **argv);
bool CheckCommonOptions();

void SetTagLogStream(const char *psz_tag, std::ostream* stream);
void SetTagLogStream(const char *psz_tag, const std::string & filename);

// 
template<typename T>
void FillWithZero(std::vector<T> &x)
{
	if (x.size() != 0) memset(&x[0], 0, sizeof(x[0]) * x.size());
}

template<typename T>
inline typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value,
    std::string>::type
    to_string(T const val) {
    return std::to_string(static_cast<long long>(val));
}

template<typename T>
inline typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value,
    std::string>::type
    to_string(T const val) {
    return std::to_string(static_cast<unsigned long long>(val));
}

template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, std::string>::type
to_string(T const val) {
        return std::to_string(static_cast<long double>(val));
}


// Errors, Warnings, Messages
#define ERROR(str) throw MlipException((std::string)"ERROR: "  + str + \
" \n       Thrown by function " + __FUNCTION__ + \
",\n       source file: " + __FILE__ + \
",\n       line: " + std::to_string(static_cast<long long>(__LINE__)) + "\n") 
void Warning(const std::string& str);
void Message(const std::string& str);
// Changes output stream for Message() and Warning(). If os==nullptr all warnings and messages will be suppressed. Returns pointer to previos stream
std::ostream* SetStreamForOutput(std::ostream* _p_os);


class MlipException {
public:
	std::string message;

	MlipException(const std::string& _message) : message(_message) {}

	virtual const char* What() const {
		return message.c_str();
	}
};
// class MlipInputException : public MlipException {
// public:
// 	MlipInputException(const std::string& _message)
// 		: MlipException(_message) {}
// };

//! escapes the string, overwriting the input. Returns the reference to the input string
std::string& mlip_string_escape(std::string& s);
//! unescapes the string, overwriting the input. Returns the reference to the input string
std::string& mlip_string_unescape(std::string& s);

//! escapes the string, overwriting the input. Returns the reference to the input string
std::string& json_string_escape(std::string& s);
//! unescapes the string, overwriting the input. Returns the reference to the input string
std::string& json_string_unescape(std::string& s);

//! parses options from a command line
void ParseOptions(int _argc, char *_argv[], std::vector<std::string>& args,
	std::map<std::string, std::string>& opts);

class Settings {
private:
	typedef std::map<std::string, std::string>::iterator Iterator;
	typedef std::map<std::string, std::string>::const_iterator ConstIterator;
	std::map<std::string, std::string> settings;
	static const std::string EMPTY_STRING;
public:
	const std::string& operator [](const std::string& key) const {
		auto pos = settings.find(key);
		if (pos == settings.end()) return EMPTY_STRING;
		return pos->second;
	}
	std::string& operator [](const std::string& key) {
		return settings[key];
	}
	//Iterator begin() { return settings.begin(); }
	//ConstIterator begin() const { return settings.begin(); }
	//Iterator end() { return settings.end(); }
	//ConstIterator end() const { return settings.end(); }

	void Load(const std::string& filename);
	void Modify(std::map<std::string, std::string>& opts);
	friend Settings LoadSettings(const std::string& filename);
	Settings ExtractSubSettings(std::string key) const;

	Settings(std::map<std::string, std::string>&& _settings)
		: settings(std::move(_settings)) {}
	Settings(const std::map<std::string, std::string>& _settings)
		: settings(_settings) {}
	Settings() {}
};

Settings LoadSettings(const std::string& filename);

namespace MLP_LEGACY_NS 
{

//! class for writing logs (to be inherited by other classes) 
class LogWriting
{
private:
	class NullBuffer : public std::streambuf
	{
		public: int overflow(int c) { return c; }
	};
	NullBuffer null_buffer;
	std::ostream null_stream;

	bool delete_p_logstream = false;

protected:
	std::ostream* p_logstream = nullptr;

public:
	LogWriting() : null_stream(&null_buffer) {};
	LogWriting(std::ostream* p_os);
	LogWriting(const std::string& filename);
	void SetLogStream(std::ostream* p_os);
	void SetLogStream(const std::string& filename);
	std::ostream* GetLogStream();
	~LogWriting();

#define logstrm if (p_logstream == nullptr); else *p_logstream 
};

} // namespace MLP_LEGACY_NS

class InitBySettings
{
private:
	std::map<std::string, bool*>		connector_bool;
	std::map<std::string, int*>			connector_int;
	std::map<std::string, double*>		connector_double;
	std::map<std::string, std::string*> connector_string;

protected:
	void MakeSetting(bool& parameter, const std::string& setting_name);
	void MakeSetting(int& parameter, const std::string& setting_name);
	void MakeSetting(double& parameter, const std::string& setting_name);
	void MakeSetting(std::string& parameter, const std::string& setting_name);

	void ApplySettings(const Settings& settings);

//	virtual void InitSettings() = 0; // to be redefined if there are bool/int/... settings in the base class

//	InitBySettings()
//	{
//		InitSettings();
//	}

	void PrintSettings();
};
//
//class MyOstream : public std::ostream
//{
//	MyOstream(const std::string& out)
//		std::ostream()
//	{
//		
//	};
//
//
//};

#endif //MLIP_UTILS_H
