

#ifndef DEF_STRING_HELPER
	#define DEF_STRING_HELPER

	#pragma message ( "   Include: string_helper" )

	#include "common_defs.h"
	#include <string>
	#include <vector>

	#ifndef DEF_OBJTYPE
		typedef	uint32_t		objType;
	#endif
	std::string strFilebase ( std::string str );	// basename of a file (minus ext)
	std::string strFilepath ( std::string str );	// path of a file

	// convert
	bool isFloat (std::string s);	// fast	
	int strToI (std::string s);
	float strToF (std::string s);
	double strToD (std::string s);
	float strToDateF( std::string s, int mp=0, int mc=2, int dp=3, int dc=2, int yp=6, int yc=4 );
	void strFromDateF ( float f, int& m, int& d, int& y );
	unsigned char strToC ( std::string s );	
	unsigned long strToUL ( std::string s );		
	unsigned long strToID ( std::string str );		// should only be used for 4-byte literals. for actual unsigned long see strToUL
	bool strToVec ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec, int cpt=3);	
	bool strToVec3 ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec );	
	bool strToVec4 ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec );	
	std::string cToStr ( char c );
	std::string iToStr ( int i );
	std::string fToStr ( float f );	
	std::string xlToStr ( uint64_t v );	
	objType strToType ( std::string str );
	std::string typeToStr ( objType t );
	std::string wsToStr ( const std::wstring& str );
	std::wstring strToWs (const std::string& s);	

	//--------------------- BETTER NAMED API	
	bool		strSplit ( std::string str, std::string sep, std::string& left, std::string& right );		// "object,more".. left='object', right='more', str=unchanged
	std::string strSplitLeft ( std::string& str, std::string sep );											// "object,more".. str='more', return='object'
	std::string strSplitRight ( std::string& str, std::string sep );										// "object:more".. str='object', return='more'	
	int			strSplitMultiple ( std::string str, std::string sep, std::vector<std::string>& list );		// obj1,obj2,obj3.. list={ob1, obj2, obj3}
	bool		strFileSplit ( std::string str, std::string& path, std::string& name, std::string& ext );	

	bool		strParseKeyVal ( std::string& str, std::string lsep, std::string rsep, std::string& key, std::string& val ); // "object<car>,other".. str='other', key='object', val='car'


	//----------- original api

	std::string strParseOut ( std::string& str, std::string sep, std::string whitespace );					// e.g. "date=VEC4 | more" --> result "VEC4", str="date | more" 
	std::string strParse ( std::string str, std::string lstr, std::string rstr, std::string lsep, std::string rsep );
	std::string strParseArg ( std::string& tag, std::string valsep, std::string sep, std::string& str );
	std::string strParseFirst ( std::string& str, std::string sep, std::string others, char& ch );
	
	bool		strGet ( std::string str, std::string& result, std::string lsep, std::string rsep );	
	bool		strGet ( const std::string& s, std::string lsep, std::string rsep, std::string& result, size_t& pos );		
	
	bool		strSub ( std::string str, int first, int cnt, std::string cmp );	
	std::string strReplace ( std::string str, std::string src, std::string dest );
	bool		strReplace ( std::string& str, std::string src, std::string dest, int& cnt );
	int			strExtract ( std::string& str, std::vector<std::string>& list );
	int			strFindFromList ( std::string str, std::vector<std::string>& list, int& pos );
	bool		strEmpty ( const std::string& s);	

	// trimmin
	std::string strLTrim ( std::string str );
	std::string strRTrim ( std::string str );
	std::string strTrim ( std::string str );
	std::string strTrim ( std::string str, std::string ch );
	std::string strLeft ( std::string str, int n );
	std::string strRight ( std::string str, int n );	
	std::string strLeftOf ( std::string str, std::string sep );
	std::string strMidOf ( std::string str, std::string sep );
	std::string strRightOf ( std::string str, std::string sep );	
	
	// alphanumeric
	bool strIsNum ( std::string str, float& f );	
	float strToNum ( std::string str );	
	int strCount ( std::string& str, char ch );		

	bool readword ( char *line, char delim, char *word, int max_size  );
	std::string readword ( char *line, char delim );

	#define MAKECLR(r,g,b,a)	( (uint(a*255.0f)<<24) | (uint(b*255.0f)<<16) | (uint(g*255.0f)<<8) | uint(r*255.0f) )
	#define ALPH(c)			(float((c>>24) & 0xFF)/255.0)
	#define BLUE(c)			(float((c>>16) & 0xFF)/255.0)
	#define GRN(c)			(float((c>>8)  & 0xFF)/255.0)
	#define RED(c)			(float( c      & 0xFF)/255.0)

#endif
