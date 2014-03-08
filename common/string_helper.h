
#ifndef DEF_STRING_HELPER
	#define DEF_STRING_HELPER

#include <vector>
#include <sstream>
#include <algorithm>

typedef std::vector< std::string > str_vec_t;

class StringHelper {
public:
	static std::vector< std::string > &SplitString( const std::string &s, char delim, std::vector< std::string > &elems ) {
		std::stringstream ss(s);
		std::string item;
		while(std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
		return elems;
	}

	static std::vector< std::string > SplitString( std::vector< std::string > elems, char delim ) {
		std::stringstream ss;
		std::string subDelim;
		ss << delim;
		ss >> subDelim;

		std::vector< std::string > out;

		for ( int i=0; i < (int) elems.size() - 1; ++i ){
			if ( elems.size() > 0 ) {
				std::vector< std::string > subVec;
				subVec = SplitString( elems[i], delim );

				if ( subVec.size() > 1 ) { 
					for ( int t=0; t < (int) subVec.size() - 1; ++t ){
						if ( subVec[t] != subDelim ) { out.push_back(subVec[t]); }
						out.push_back( subDelim );
					}
				}
				if ( subVec.size() > 0 ) {
					out.push_back(subVec[subVec.size() - 1]);				
				}
			}
		}

		return out;

	}

	static std::vector< std::string > SplitString( const std::string &s, char delim ) {
		std::vector< std::string > elems;
		return SplitString(s, delim, elems);
	}

	static std::vector<std::string> SplitString(const std::string &s, const std::string& delim, const bool keep_empty = true) {
		std::vector<std::string> result;
		if (delim.empty()) {
			result.push_back(s);
			return result;
		}
		std::string::const_iterator substart = s.begin(), subend;
		while (true) {
			subend = search(substart, s.end(), delim.begin(), delim.end());
			std::string temp(substart, subend);
			if (keep_empty || !temp.empty()) {
				result.push_back(temp);
			}
			if (subend == s.end()) {
				break;
			}
			substart = subend + delim.size();
		}
		return result;
	}

	static std::string Replace ( const std::string &s, const std::string& target, const std::string& replacement ) {

		std::string out = "";

		std::vector<std::string> split = StringHelper::SplitString( s, target );
		for ( int i=0; i < (int) split.size() - 1; ++i ){
			out += split[i] + replacement;
		}

		out += split[split.size() - 1];

		return out;
	}


	static std::string FileSanitizer( const std::string s ) {
		
		//TODO important that this is complete.  review with others

		std::string trans = s;

		std::string targets = " ,@,!,*,^,$,%,#,(,),+,;,:,>,<,?,/,\\,[,],{,}";
		std::vector< std::string > targetSet = SplitString( targets, ',' );
		targetSet.push_back(","); // must account for the comma!

		for ( int t=0; t < (int) targetSet.size(); t++ ){
			int pos = (int) trans.find( targetSet[t] );
			while ( pos != std::string::npos ) {
				trans.replace( pos, 1, "_" );
				pos = (int) trans.find( targetSet[t], pos + 1 );
			} 
		}

		return trans;
	}

	static bool IsNumeric( const std::string s ){
		std::string trans = s;

		std::string targets = "0,1,2,3,4,5,6,7,8,9";
		std::vector< std::string > targetSet = SplitString( targets, ',' );
		//targetSet.push_back(","); // must account for the comma!

		for ( unsigned int n=0; n < trans.length(); n++ ) {
			int pos = (int) targets.find( trans[n] );
			if ( pos == std::string::npos ) { return false; }
		}

		return true;
	}

private:
	StringHelper();

};

#endif