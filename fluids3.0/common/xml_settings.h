

#ifndef DEF_XML_SETTINGS
	#define DEF_XML_SETTINGS

	#include <string>
	#include <map>
	
	#include "app_util.h"
	
	#define XML_BADVAL		-65535

	class TiXmlHandle;
	class TiXmlNode;
	class TiXmlDocument;
	class TiXmlElement;

	class XmlSettings {
		public:
			XmlSettings();
			~XmlSettings ();
			bool Load ( std::string filename ) { return Load( filename, "" ); }
			bool Load ( std::string filename, std::string section );

			bool hasAttribute ( std::string key );
			int numAttribute ( std::string key );

			void assignValueF ( float* buf, std::string key );
			void assignValueD ( double* buf, std::string key );
			void assignValueV3 ( Vector3DF* buf, std::string key );
			void assignValueStr ( std::string& str, std::string key );
			
			bool setBase ( std::string name, int id );

			TiXmlElement* FindElement ( int n, std::string key );

			//TODO this interface doesn't really allow for the full hierarchical expression to be realized, 
			//but will do for our immediate purposes.
			float getValueF ( std::string key );
			int getValueI ( std::string key );
			std::string getValueStr ( std::string key );
			unsigned char getValueUChar ( std::string key );
			
			// Get a key-values relative to the base node
			float getAttributeF ( int n, std::string key, std::string attrib );
			int getAttributeI ( int n, std::string key, std::string attrib );
			std::string getAttributeStr ( int n, std::string key, std::string attrib );

			// Get parts of a specific node
			TiXmlNode* getChild ( std::string name );
			TiXmlNode* getChild ( TiXmlNode* node, std::string name );
			float getAttributeF ( TiXmlElement* node, std::string attrib );
			char getAttributeC ( TiXmlElement* node, std::string attrib );
			int getAttributeI ( TiXmlElement* node, std::string attrib );
			std::string getAttributeStr ( TiXmlElement* node, std::string attrib );
			std::string getContent ( TiXmlNode* node );

			//Utility
			static Vector3DF ParseVector3 ( std::string vec );
			static Vector4DF ParseVector4 ( std::string vec );

		private:
			TiXmlHandle*			mRoot;
			TiXmlHandle*			mCurrBase;
			TiXmlDocument*			mDocument;

	};

#endif