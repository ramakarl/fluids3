
#include "xml_settings.h"
#include "string_helper.h"

#pragma message ( "   Include: tinyxml.h (TinyXml)" )
#include "tinyxml.h"


XmlSettings::XmlSettings()
{ 
	mDocument = 0x0;
	mCurrBase = 0x0;
	mRoot = 0x0;
}

XmlSettings::~XmlSettings ()
{
	if ( mDocument != 0x0 ) { delete mDocument; }
	if ( mCurrBase != 0x0 ) { delete mCurrBase; }
	if ( mRoot != 0x0 ) { delete mRoot; }
}

bool XmlSettings::Load( std::string filename, std::string section )
{
	if ( mDocument != 0x0 ) { delete mDocument; }
	if ( mCurrBase != 0x0 ) { delete mCurrBase; }
	if ( mRoot != 0x0 ) { delete mRoot; }

	mDocument = new TiXmlDocument( filename );
	if (!mDocument->LoadFile()) return false;

	TiXmlHandle hDoc(mDocument);
	TiXmlElement* pElem;

	// Get first element
	pElem = hDoc.FirstChildElement().Element();	
	if (!pElem) return false;	

	// Maintain the root node of document
	mRoot = new TiXmlHandle( pElem );

	// Maintain a node at some arbitrary point in doc
	mCurrBase = new TiXmlHandle( 0 );

	if ( section.length() > 0 ) {
		if ( section.compare ( mRoot->Node()->Value() ) == 0 ) {
			*mCurrBase = *mRoot;			// Set base equal to root (section we want is root)
		} else {
											// Set base to a child of root
			*mCurrBase = (*mRoot).FirstChild( section ); 
		}
	} else {
		*mCurrBase = *mRoot;				// No section selected. Set base to root.
	}
	if ( mCurrBase == 0x0 ) return false;
	return true;
}

bool XmlSettings::setBase ( std::string name, int id )
{
	TiXmlElement* elem = (*mRoot).ToElement();
	int vid = 0;
	while ( elem  != 0x0 && vid != id) {		
		elem = elem->NextSiblingElement();
		if ( elem->Attribute ( "id" ) != 0x0 && elem->ValueStr().compare(name)==0 ) 
			vid = strToI ( elem->Attribute ( "id" ) );
	}
	if ( elem == 0x0 ) elem = (*mRoot).ToElement();

	if ( mCurrBase != 0x0 ) delete mCurrBase;
	mCurrBase = new TiXmlHandle ( elem );

	return true;
}

TiXmlNode* XmlSettings::getChild ( std::string name )
{
	return (*mCurrBase).FirstChild ( name ).Node();
}
TiXmlNode* XmlSettings::getChild ( TiXmlNode* node, std::string name )
{
	return node->FirstChild ( name );
}

void XmlSettings::assignValueF ( float* buf, std::string key )
{	
	TiXmlElement* elem = (*mCurrBase).ToNode()->FirstChildElement( key );
	if ( elem != 0x0 && elem->GetText() != 0x0 ) {
		*buf = strToF ( elem->GetText() );		// Assignment only occurs if xml element exists
	}
}
void XmlSettings::assignValueD ( double* buf, std::string key )
{	
	TiXmlElement* elem = (*mCurrBase).ToNode()->FirstChildElement( key );
	if ( elem != 0x0 && elem->GetText() != 0x0 ) {
		*buf = strToF ( elem->GetText() );		// Assignment only occurs if xml element exists
	}
}
void XmlSettings::assignValueV3 ( Vector3DF* buf, std::string key )
{	
	TiXmlElement* elem = (*mCurrBase).ToNode()->FirstChildElement( key );
	if ( elem != 0x0 && elem->GetText() != 0x0 ) {
		*buf = ParseVector3 ( elem->GetText() );	// Assignment only occurs if xml element exists
	}
}
void XmlSettings::assignValueStr( std::string& str, std::string key )
{	
	str = getValueStr ( key );
}


std::string XmlSettings::getValueStr ( std::string key )	
{
	TiXmlElement* elem = (*mCurrBase).ToNode()->FirstChildElement( key );
	if ( elem != 0x0 && elem->GetText() != 0x0 ) {
		return elem->GetText();
	}
	return "";
}

float XmlSettings::getValueF ( std::string key )
{
	std::string vstr = getValueStr ( key );
	if ( vstr.length()==0 ) return XML_BADVAL;
	return strToF ( vstr );
}


int XmlSettings::getValueI ( std::string key )
{
	std::string vstr = getValueStr ( key );
	if ( vstr.length()==0 ) return XML_BADVAL;
	return strToI ( vstr );
}

unsigned char XmlSettings::getValueUChar( std::string key )
{
	std::string s = getValueStr( key );
	if ( s.size() == 1 )
		return strToC ( s );
	else 
		return (unsigned char) strToI ( s );
}

bool XmlSettings::hasAttribute ( std::string key )
{
	TiXmlElement* child = (*mCurrBase).FirstChildElement( key ).ToElement();
	if ( child == 0x0 ) return false;
	return true;
}

int XmlSettings::numAttribute ( std::string key )
{
	int num = 0;
	TiXmlElement* child = (*mCurrBase).FirstChildElement( key ).ToElement();
	while ( child != 0x0 ) {
		child = child->NextSiblingElement( key );
		num++;
	}
	return num;
}

TiXmlElement* XmlSettings::FindElement ( int n, std::string key )
{
	TiXmlElement* child = (*mCurrBase).FirstChildElement( key ).ToElement();
	while ( n > 0 && child != 0x0 ) {
		child = child->NextSiblingElement();
		n--;
	}
	return child;
}

std::string XmlSettings::getAttributeStr ( int n, std::string key, std::string attrib )	
{
	return getAttributeStr ( FindElement ( n, key ), attrib );
}
std::string XmlSettings::getAttributeStr ( TiXmlElement* elem, std::string attrib )
{
	if ( elem == 0x0 ) return "";
	if ( elem->Attribute ( attrib.c_str() ) == 0x0 ) return "";
	return elem->Attribute ( attrib.c_str() );
}

float XmlSettings::getAttributeF ( int n, std::string key, std::string attrib )
{
	return getAttributeF ( FindElement ( n, key ), attrib );
}
float XmlSettings::getAttributeF ( TiXmlElement* elem, std::string attrib )
{
	if ( elem == 0x0 ) return XML_BADVAL;
	std::string vstr = "";
	if ( elem->Attribute ( attrib.c_str() ) != 0x0 ) vstr = elem->Attribute ( attrib.c_str() );
	if ( vstr.length()==0 ) return XML_BADVAL;
	return strToF ( vstr );
}

int XmlSettings::getAttributeI ( int n, std::string key, std::string attrib )
{
	return getAttributeI ( FindElement ( n, key ), attrib );
}
int XmlSettings::getAttributeI ( TiXmlElement* elem, std::string attrib )
{
	if ( elem == 0x0 ) return XML_BADVAL;
	std::string vstr = "";
	if ( elem->Attribute ( attrib.c_str() ) != 0x0 ) vstr = elem->Attribute ( attrib.c_str() );
	if ( vstr.length()==0 ) return XML_BADVAL;
	return strToI ( vstr );
}


char XmlSettings::getAttributeC ( TiXmlElement* elem, std::string attrib )
{
	if ( elem == 0x0 ) return (char) XML_BADVAL;
	std::string vstr = "";
	if ( elem->Attribute ( attrib.c_str() ) != 0x0 ) vstr = elem->Attribute ( attrib.c_str() );
	return vstr[0];
}

std::string XmlSettings::getContent ( TiXmlNode* node )
{
	return node->FirstChild()->Value();
}

Vector3DF XmlSettings::ParseVector3 ( std::string vec )
{
	std::vector< std::string > vals = StringHelper::SplitString( vec, ',' );	
	float x = atof( vals[0].c_str() );
	float y = atof( vals[1].c_str() );
	float z = atof( vals[2].c_str() );
	return Vector3DF( x, y, z );
}

Vector4DF XmlSettings::ParseVector4 ( std::string vec )
{
	if ( vec.length()==0 ) return Vector4DF();
	str_vec_t vals = StringHelper::SplitString( vec, ',' );	

	float x = atof( vals[0].c_str() );
	float y = atof( vals[1].c_str() );
	float z = atof( vals[2].c_str() );
	float w = atof( vals[3].c_str() );
	return Vector4DF( x, y, z, w );

}