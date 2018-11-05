#ifndef FILE_PARSER_H
#define FILE_PARSER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <ctype.h>

class FileParserBase {
public:
	FileParserBase() {}
	virtual ~FileParserBase() {}

public:
	virtual bool OpenFile(const char* path) = 0;
	virtual bool CloseFile() = 0;

	virtual bool ReadSample(double& y, std::vector<std::pair<size_t, double> >& x) = 0;

public:
	static bool FileExists(const char* path);
};

// FileParser: parse training file with LIBSVM-format
class FileParser : public FileParserBase {
public:
	FileParser();
	virtual ~FileParser();

	virtual bool OpenFile(const char* path);
	virtual bool CloseFile();

	// Read a new line and Parse to <x, y>, thread-safe but not optimized for multi-threading
	virtual bool ReadSample(double& y, std::vector<std::pair<size_t, double> >& x);

	bool ParseSample(char* buf, double& y,
		std::vector<std::pair<size_t, double> >& x);

	// Read a new line using external buffer
	char* ReadLine(char *buf, size_t& buf_size);

private:
	// Read a new line using internal buffer and copy that to allocated new memory
	char* ReadLine();

	char* ReadLineImpl(char *buf, size_t& buf_size);

private:
	enum { kDefaultBufSize = 10240 };

	FILE* file_desc_;
	char* buf_;
	size_t buf_size_;

};

template<typename T>
T* alloc_func(size_t size) {
	void* ptr = malloc(size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}

template<typename T>
T* realloc_func(T* buf, size_t size) {
	void* ptr = realloc(reinterpret_cast<void*>(buf), size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}


bool FileParserBase::FileExists(const char* path) {
	FILE *fp = fopen(path, "r");
	if (fp) {
		fclose(fp);
		return true;
	}

	return false;
}


FileParser::FileParser() : file_desc_(NULL), buf_(NULL), buf_size_(0) {
	buf_size_ = kDefaultBufSize;
	buf_ = alloc_func<char>(buf_size_);
}

FileParser::~FileParser() {
	if (file_desc_) {
		fclose(file_desc_);
	}

	if (buf_) {
		free(buf_);
	}

	buf_size_ = 0;
}

bool FileParser::OpenFile(const char* path) {
    if (strcmp(path, "stdin") == 0) {
        file_desc_ = stdin;
    }
    else {
	    file_desc_ = fopen(path, "r");
    }

	if (!file_desc_) {
		return false;
	}

	return true;
}

bool FileParser::CloseFile() {
	if (file_desc_) {
		fclose(file_desc_);
		file_desc_ = NULL;
	}

	return true;
}

char* FileParser::ReadLineImpl(char* buf, size_t& buf_size) {
	if (!file_desc_) {
		return NULL;
	}

	if (fgets(buf, buf_size - 1, file_desc_) == NULL) {
		return NULL;
	}

	while (strrchr(buf, '\n') == NULL) {
		buf_size *= 2;
		buf = realloc_func<char>(buf, buf_size);
		size_t len = strlen(buf);
		if (fgets(buf + len, buf_size - len - 1, file_desc_) == NULL) break;
	}

	return buf;
}

char* FileParser::ReadLine() {

	char *buf = ReadLineImpl(buf_, buf_size_);
	if (buf) {
		buf_ = buf;
		return strdup(buf);
	}

	return NULL;
}

char* FileParser::ReadLine(char *buf, size_t& buf_size) {
	return ReadLineImpl(buf, buf_size);
}

template<typename T>
T string_to_real(const char *nptr, char **endptr);

template<>
float string_to_real<float> (const char *nptr, char **endptr) {
	return strtof(nptr, endptr);
}

template<>
double string_to_real<double> (const char *nptr, char **endptr) {
	return strtod(nptr, endptr);
}

bool FileParser::ParseSample(char* buf, double& y,
		std::vector<std::pair<size_t, double> >& x) {
	if (buf == NULL) return false;

	char *endptr, *ptr;
	char *p = strtok_r(buf, " \t\n", &ptr);
	if (p == NULL) return false;

	y = string_to_real<double> (p, &endptr);
	if (endptr == p || *endptr != '\0') return false;
	if (y < 0) y = 0;

	x.clear();
	// add bias term
	x.push_back(std::make_pair((size_t)0, (double)1));
	while (1) {
		char *idx = strtok_r(NULL, ":", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10);
		if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
			error_found = true;
		}

		double v = string_to_real<double> (val, &endptr);
		if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
			error_found = true;
		}

		if (!error_found) {
			x.push_back(std::make_pair(k, v));
		}
	}

	return true;
}

bool FileParser::ReadSample(double& y,
		std::vector<std::pair<size_t, double> >& x) {
	char *buf = ReadLineImpl(buf_, buf_size_);
	if (!buf) return false;

	buf_ = buf;
	return ParseSample(buf, y, x);
}


#endif // SRC_FILE_PARSER_H
