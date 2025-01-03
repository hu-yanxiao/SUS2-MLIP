/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#include <iostream>
#include <string>
#include "utils.h"
#include <stdarg.h>  
#ifdef MLIP_MPI
#include <mpi.h>
#endif

using namespace std;

#ifdef _MSC_VER
#pragma warning (disable: 4996) // disable unsafe functions warnings in VC
#endif

enum EOptions { ELOG_KEYS=0, ELOG_FILE, EOPTIONS_SIZE};
enum EOptionType { EOPTION_INT, EOPTION_DOUBLE, EOPTOION_STR };

struct OptionItem
{
   const char *short_name;
   const char *long_name;
   EOptionType type;
};

OptionItem options[] = {
                          {"-lk","--logkeys",EOPTOION_STR},          // ELOG_KEYS
                          {"-lf","--logfile",EOPTOION_STR}           // ELOG_FILE
                       };
std::map<int,void*> option_value;

// Print usage statement
// @TODO: move this to mlp stuff
void Usage()
{
  printf("Usage: mlp [COMMON OPTION] COMMAND [COMMAND OPTION]\n\n"
         "COMMON OPTIONS:\n"
         "\t-h, --help\t\t\t\t: help statement\n"
         "\t-lk | --logkeys <tag>,..\t\t: the tags to be output in logging\n"
         "\t-lf | --logfile <filename>\t\t: the logging file name\n\n"
         "See help command for more details.\n"
         );
}

// Print a string to stdout or to the file
// input:
//      psz_tag - tag name
//      psz_str - string to be output
void OutputString(const char* psz_tag, const char* psz_str)
{
	char sz_buff[512];
	char *psz_logkeys = (char*)option_value[ELOG_KEYS];
	if (NULL == psz_logkeys) return;
	strncpy(sz_buff, psz_logkeys, 512);
	char *psz = strtok(sz_buff, ",;");
	while (psz) {
		if (0 == strcmp(psz, psz_tag) || 0 == strcmp(psz_logkeys, "**")) {
			char *psz_logfile = (char*)option_value[ELOG_FILE];
			if (NULL == psz_logfile)
				fprintf(stdout, psz_str);
			else {
				FILE *file = fopen(psz_logfile, "a+");
				fprintf(file, psz_str);
				fclose(file);
			}
		}
		psz = strtok(NULL, ",;");
	}
}

// Composes a string to be passed to the function OutputString
// input:
//      psz_tag - tag name
//      psz_format - the string that contains a format string follow
//          specifications of prinf function
void LogOutput(const char* psz_tag, const char* psz_format, ...)
{
	char sz_buff[1000];
	sprintf(sz_buff, "[%s] ", psz_tag);
	va_list args;
	va_start(args, psz_format);
	vsprintf(sz_buff + strlen(sz_buff), psz_format, args);

	OutputString(psz_tag, sz_buff);

	va_end(args);
}

// The collection of file streams used for logging
std::map<const char *, std::ofstream> map_logfiles;

void SetTagLogStream(const char *psz_tag, std::ostream* stream)
{
	std::ofstream& buff = map_logfiles[psz_tag];
	buff.basic_ios<char>::rdbuf(stream->rdbuf());
}


void SetTagLogStream(const char *psz_tag, const std::string & filename)
{
	if (filename.empty()) {
	} else if (filename == "stdout") {
		std::ofstream& buff = map_logfiles[psz_tag];
		buff.basic_ios<char>::rdbuf(std::cout.rdbuf());
	} else if (filename == "stderr") {
		std::ofstream& buff = map_logfiles[psz_tag];
		buff.basic_ios<char>::rdbuf(std::cerr.rdbuf());
	} else {
		std::ofstream& buff = map_logfiles[psz_tag];
		if (buff.is_open() == false ) {
        remove(filename.c_str());
        buff.open(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        if (!buff.is_open())
			      ERROR("Can't open file \"" + filename + "\" for logging");
    }
	}
}

// Print a string to output file stream
// input: 
//      psz_tag - tag name
//      str - string to be output
void LogOutput(const char* psz_tag, const std::string& str)
{
	for (auto& istream : map_logfiles) {
		if (0 == strcmp(istream.first, psz_tag))
			istream.second << str;
		istream.second.flush();
	}

	//   if( map_logfiles.find(psz_tag) != mapstream.end() )
	//   {
	//     map_logfiles[psz_tag] << str;
	//     map_logfiles[psz_tag].flush();
	//   } 
}

// Parse command line optons
//  input  : argc, argv values.
//  output : the next index of argument after parsing of common options,
//           negative if something wrong.
int ParseCommonOptions(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; ) {
		if (argv[i][0] != '-') return i;
		if (0 == strcmp(argv[i], "-h") || 0 == strcmp(argv[i], "--help")) {
			Usage();
			return -1;
		}
		for (int j = 0; true; j++) {
			if (j >= EOPTIONS_SIZE) {
				Warning("Unrecognized option is specified.");
				return -4;
			}
			if (0 == strcmp(argv[i], options[j].short_name) ||
				0 == strcmp(argv[i], options[j].long_name)) {
				if (option_value.find(j) != option_value.end()) {
					Warning("Option is specified multiple times.");
					return -3;
				}
				++i;
				if (i >= argc) {
					Warning("Missing argument for the option.");
					return -2;
				}
				switch ((int)options[j].type) {
				case EOPTOION_STR:
					option_value.insert(std::pair<int, void*>(j, (void*)argv[i]));
					break;
				} // switch (type)
				++i;
				break;
			}
		} // loop j
	} // loop i
	return i;
} //

// Check loaded common option 
//  output : false if something wrong
bool CheckCommonOptions()
{
	bool res = true;
	char *psz_logfile = (char*)option_value[ELOG_FILE];
	if (NULL != psz_logfile) {
		FILE *file = fopen(psz_logfile, "w");
		if (NULL == file)
			res = false, Warning("Logging file openned with failure.");
		else
			fclose(file);
	}
	return res;
}


//#ifdef _MSC_VER
//#pragma warning (disable: 4996) // disable unsafe functions warnings in VC
//#endif


std::ostream* p_os = &std::cout;


void Warning(const std::string& str)
{
	//std::cerr << "WARNING: " << str << std::endl;
	if (p_os != nullptr)
		*p_os << "MLIP WARNING: " << str << std::endl;
}

void Message(const std::string& str)
{
	if (p_os != nullptr)
		*p_os << "MLIP: " << str << std::endl;
}

std::ostream* SetStreamForOutput(std::ostream* _p_os)
{
	std::ostream* tmp = p_os;
	p_os = _p_os;
	return tmp;
}


std::string& mlip_string_escape(std::string& s) {
	std::size_t found = (std::size_t)-1;

	while ((found = s.find_first_of(" \t\n\r\\", found + 1)) != std::string::npos) {
		if (s[found] == ' ') s.replace(found, 1, "\\_");
		else if (s[found] == '\t') s.replace(found, 1, "\\t");
		else if (s[found] == '\n') s.replace(found, 1, "\\n");
		else if (s[found] == '\r') s.replace(found, 1, "\\r");
	}
	return s;
}

std::string& mlip_string_unescape(std::string& s) {
	std::size_t found = (std::size_t)-1;

	while ((found = s.find_first_of("\\", found + 1)) != std::string::npos) {
		if (s[found + 1] == '_') s.replace(found, 2, " ");
		else if (s[found + 1] == 't') s.replace(found, 2, "\t");
		else if (s[found + 1] == 'n') s.replace(found, 2, "\n");
		else if (s[found + 1] == 'r') s.replace(found, 2, "\r");
	}
	return s;
}

std::string& json_string_escape(std::string& s) {
	std::size_t found = (std::size_t)-1;

	while ((found = s.find_first_of(" \t\n\r\\\"", found + 1)) != std::string::npos) {
		if (s[found] == ' ') s.replace(found, 1, "\\_");
		else if (s[found] == '\t') s.replace(found, 1, "\\t");
		else if (s[found] == '\n') s.replace(found, 1, "\\n");
		else if (s[found] == '\r') s.replace(found, 1, "\\r");
		else if (s[found] == '\"') s.replace(found, 1, "\\\"");
	}
	return s;
}

std::string& json_string_unescape(std::string& s) {
	std::size_t found = (std::size_t)-1;

	while ((found = s.find_first_of("\\", found + 1)) != std::string::npos) {
		if (s[found + 1] == '_') s.replace(found, 2, " ");
		else if (s[found + 1] == 't') s.replace(found, 2, "\t");
		else if (s[found + 1] == 'n') s.replace(found, 2, "\n");
		else if (s[found + 1] == 'r') s.replace(found, 2, "\r");
		else if (s[found + 1] == '\"') s.replace(found, 2, "\"");
	}
	return s;
}

namespace MLP_LEGACY_NS {

LogWriting::LogWriting(std::ostream * _p_os) : null_stream(&null_buffer)
{
	SetLogStream(_p_os);
}

LogWriting::LogWriting(const std::string & filename) : null_stream(&null_buffer)
{
	SetLogStream(filename);
}

void LogWriting::SetLogStream(std::ostream * _p_os)
{
	if (delete_p_logstream)
		delete p_logstream;
	p_logstream = _p_os;
}

void LogWriting::SetLogStream(const std::string & filename)
{
	if (delete_p_logstream)
		delete p_logstream;
	if (filename.empty())
		p_logstream = nullptr;
	else if (filename == "stdout")
		SetLogStream(&std::cout);
	else if (filename == "stderr")
		SetLogStream(&std::cerr);
	else
	{
		p_logstream = new std::ofstream(filename);
		if (!((std::ofstream*)p_logstream)->is_open())
			ERROR("Can't open file \"" + filename + "\" for logging");
		delete_p_logstream = true;
	}
}

std::ostream * LogWriting::GetLogStream()
{
	return p_logstream;
}

LogWriting::~LogWriting()
{
	if (delete_p_logstream)
		delete p_logstream;
}

} // namespace MLP_LEGACY_NS

const std::string Settings::EMPTY_STRING = "";

void Settings::Load(const string& filename)
{
	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Cannot open the settings file " + filename);
	else
	{
		//Message("Reading settings from " + filename);

		string buff_key;
		ifs >> buff_key;
		while (!ifs.eof()) {
			if (buff_key.substr(0, 1) != "#") {
				string buff_val = "";
				ifs >> buff_val;
				if (buff_val.substr(0, 1) != "#")
					settings[buff_key] = buff_val;
			}

			ifs.ignore(HUGE_INT, '\n');
			ifs >> buff_key;
		}
	}
}

void Settings::Modify(map<string, string>& opts)
{
	for (auto setting : opts)
		settings[setting.first] = setting.second;
}

Settings LoadSettings(const std::string& filename)
{
	Settings settings;
	settings.Load(filename);
	return settings;
}

Settings Settings::ExtractSubSettings(string key) const
{
	key += ":";
	Settings result;
	for (auto setting : settings) {
		if (setting.first.rfind(key, 0) == 0) // if beginning matches
			result.settings.emplace(setting);
	}
	return result;
}


void InitBySettings::MakeSetting(bool & parameter, const std::string & setting_name)
{
	connector_bool.emplace(setting_name, &parameter);
}

void InitBySettings::MakeSetting(int & parameter, const std::string & setting_name)
{
	connector_int.emplace(setting_name, &parameter);
}

void InitBySettings::MakeSetting(double & parameter, const std::string & setting_name)
{
	connector_double.emplace(setting_name, &parameter);
}

void InitBySettings::MakeSetting(std::string & parameter, const std::string & setting_name)
{
	connector_string.emplace(setting_name, &parameter);
}

void InitBySettings::ApplySettings(const Settings& settings)
{
	for (auto setting : connector_bool) {
		// we have to play with constant iterators to entries of settings
		if (settings[setting.first] != "") {
			if (settings[setting.first] == "FALSE")
				*setting.second = false;
			else if (settings[setting.first] == "TRUE")
				*setting.second = true;
			else
				ERROR((std::string)"settings[\"" + setting.first + "\"]=\"" + settings[setting.first] + "\""
					" but it is not \"TRUE\" or \"FALSE\"");
		}
	}

	for (auto& setting : connector_int)
		if (settings[setting.first] != "") {
			try {
				*setting.second = stoi(settings[setting.first]);
			}
			catch (...) {
				ERROR("while parsing option " + setting.first + ": "
					+ settings[setting.first] + " is not a valid int.");

			}
		}

	for (auto& setting : connector_double)
		if (settings[setting.first] != "") {
			try {
				*setting.second = stod(settings[setting.first]);
			}
			catch (...) {
				ERROR("while parsing option " + setting.first + ": "
					+ settings[setting.first] + " is not a valid double.");
			}
		}
	for (auto& setting : connector_string)
		if (settings[setting.first] != "")
			*setting.second = settings[setting.first];
}

void InitBySettings::PrintSettings()
{
#ifdef MLIP_MPI
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	if (mpi_rank == 0)
#endif
	{
		for (auto& stng : connector_bool)
			Message('\t' + stng.first + " = " + (*stng.second ? "true" : "false"));
		for (auto& stng : connector_int)
			Message('\t' + stng.first + " = " + to_string(*stng.second));
		for (auto& stng : connector_double)
			Message('\t' + stng.first + " = " + to_string(*stng.second));
		for (auto& stng : connector_string)
			if (!stng.second->empty())
				Message('\t' + stng.first + " = " + *stng.second);
	}
}

void ParseOptions(int _argc, char *_argv[], std::vector<std::string>& args,
	std::map<std::string, std::string>& opts)
{
	bool expect_options = true;

	for (int i = 1; i < _argc; i++) 
  {
//     // parse global options
//     for (int j = 0; j < EOPTIONS_SIZE; j++)
//     {
//       if ( strcmp(_argv[i], options[j].short_name) == 0 ||
//            strcmp(_argv[i], options[j].long_name) == 0)
//       {
//          switch ((int)options[j].type)
//          {
//             case EOPTOION_STR:
//               option_value.insert(std::pair<int,void*>(j,(void*)_argv[i]));
//               break;
//          }
//         continue;
//       }
//     }
		std::string word = _argv[i];
		if (word[0] == '-' && expect_options) {
			if (word.length() == 1) ERROR("'-' is not a valid option");
			if (word[1] != '-') ERROR("option " + _argv[i] + " should start with '--', not '-'");
			if (word.length() == 2) {
				// this is '--'. Switch off expect_options
				expect_options = false;
			} else {
				size_t eq_pos = word.find_first_of("=");
				if (eq_pos == std::string::npos)
					opts[word.substr(2)] = "true";
				else
					opts[word.substr(2, eq_pos - 2)] = word.substr(eq_pos + 1);
			}
		} else
			args.push_back(_argv[i]);
	}
}
