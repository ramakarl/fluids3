// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h 

/* 
 This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Please note that the Irrlicht Engine is based in part on the work of the 
  Independent JPEG Group, the zlib and the libPng. This means that if you use
  the Irrlicht Engine in your product, you must acknowledge somewhere in your
  documentation that you've used the IJG code. It would also be nice to mention
  that you use the Irrlicht Engine, the zlib and libPng. See the README files 
  in the jpeglib, the zlib and libPng for further informations.
*/

#ifndef QUATERNION_H
	#define QUATERNION_H

	#include "vec.h"
	//#include "matrix.h"

	typedef	float	f32;
	typedef	double	f64;

	#define EPS		0.00001

	const f32 ROUNDING_ERROR_f32	= 0.000001f;
	const f64 ROUNDING_ERROR_f64	= 0.00000001;
	const f64 PI64					= 3.1415926535897932384626433832795028841971693993751;
	//const f32 PI        = 3.14159265359f;

	#define min(a,b)		( (a < b ) ? a : b )
	#define max(a,b)		( (a > b ) ? a : b )

	inline f64 clamp (f64 value, f64 low, f64 high)
	{
	   return min (max (value,low), high);
	}
	inline bool fequal ( double a, double b, double eps )
	{
		return ( fabs(a-b)<eps );
	}

	class quaternion
	{
		public:

			//! Default Constructor
			quaternion() : X(0.0f), Y(0.0f), Z(0.0f), W(1.0f) {}

			//! Constructor
			quaternion(f32 x, f32 y, f32 z, f32 w) : X(x), Y(y), Z(z), W(w) { }

			//! Constructor which converts euler angles (radians) to a quaternion
			quaternion(f32 x, f32 y, f32 z);

			//! Constructor which converts euler angles (radians) to a quaternion
			quaternion(const Vector3DF& vec);

			//! Constructor which converts a matrix to a quaternion
			quaternion(const Matrix4F& mat);

			//! Equalilty operator
			bool operator==(const quaternion& other) const;

			//! inequality operator
			bool operator!=(const quaternion& other) const;

			//! Assignment operator
			inline quaternion& operator=(const quaternion& other);

			//! Matrix assignment operator
			inline quaternion& operator=(const Matrix4F& other);

			//! Add operator
			quaternion operator+(const quaternion& other) const;

			//! Multiplication operator
			quaternion operator*(const quaternion& other) const;

			//! Multiplication operator with scalar
			quaternion operator*(f32 s) const;

			//! Multiplication operator with scalar
			quaternion& operator*=(f32 s);

			//! Multiplication operator
			Vector3DF operator*(const Vector3DF& v) const;

			//! Multiplication operator
			quaternion& operator*=(const quaternion& other);	

			//! Calculates the dot product
			inline f32 dotProduct(const quaternion& other) const;

			//! Sets new quaternion
			inline quaternion& set(f32 x, f32 y, f32 z, f32 w);

			//! Sets new quaternion based on euler angles (radians)
			inline quaternion& set(f32 x, f32 y, f32 z);

			//! Sets new quaternion based on euler angles (radians)
			inline quaternion& set(const Vector3DF& vec);

			//! Sets new quaternion from other quaternion
			inline quaternion& set(const quaternion& quat);

			//! returns if this quaternion equals the other one, taking floating point rounding errors into account
			inline bool equals(const quaternion& other,
					const f32 tolerance = ROUNDING_ERROR_f32 ) const;

			inline void fixround () {
				if ( fabs(X) < ROUNDING_ERROR_f64 ) X = 0;
				if ( fabs(Y) < ROUNDING_ERROR_f64 ) Y = 0;
				if ( fabs(Z) < ROUNDING_ERROR_f64 ) Z = 0;
				if ( fabs(W) < ROUNDING_ERROR_f64 ) W = 0;
			}

			//! Normalizes the quaternion
			inline quaternion& normalize();

			//! Creates a matrix from this quaternion
			Matrix4F getMatrix() const;

			//! Creates a matrix from this quaternion
			void getMatrix( Matrix4F &dest, const Vector3DF &translation=Vector3DF() ) const;

			/*!
				Creates a matrix from this quaternion
				Rotate about a center point
				shortcut for
				core::quaternion q;
				q.rotationFromTo ( vin[i].Normal, forward );
				q.getMatrixCenter ( lookat, center, newPos );

				core::Matrix4F m2;
				m2.setInverseTranslation ( center );
				lookat *= m2;

				core::Matrix4F m3;
				m2.setTranslation ( newPos );
				lookat *= m3;

			*/
			void getMatrixCenter( Matrix4F &dest, const Vector3DF &center, const Vector3DF &translation ) const;

			//! Creates a matrix from this quaternion
			inline void getMatrix_transposed( Matrix4F &dest ) const;

			//! Inverts this quaternion
			quaternion& makeInverse();

			// Inverse
			quaternion operator ~ ()
			{
				return quaternion(-X, -Y, -Z, W);   // W unchanged
			}
			quaternion inverse ()
			{
				return quaternion(-X, -Y, -Z, W);   // W unchanged
			}
			quaternion conjInverse ()
			{
				const f32 scale = sqrtf(X*X + Y*Y + Z*Z);
				const f32 invscale = 1.0 / scale;
	
				return quaternion(-X*invscale, -Y*invscale, -Z*invscale, W);   // W unchanged
			}



			//! Negative operator
			quaternion negative ()
			{
				return quaternion(X, Y, Z, -W);		// negative of W
			}

			//! Set this quaternion to the linear interpolation between two quaternions
			/** \param q1 First quaternion to be interpolated.
			\param q2 Second quaternion to be interpolated.
			\param time Progress of interpolation. For time=0 the result is
			q1, for time=1 the result is q2. Otherwise interpolation
			between q1 and q2.
			*/
			quaternion& lerp(quaternion q1, quaternion q2, f32 time);

			//! Set this quaternion to the result of the spherical interpolation between two quaternions
			/** \param q1 First quaternion to be interpolated.
			\param q2 Second quaternion to be interpolated.
			\param time Progress of interpolation. For time=0 the result is
			q1, for time=1 the result is q2. Otherwise interpolation
			between q1 and q2.
			\param threshold To avoid inaccuracies at the end (time=1) the
			interpolation switches to linear interpolation at some point.
			This value defines how much of the remaining interpolation will
			be calculated with lerp. Everything from 1-threshold up will be
			linear interpolation.
			*/
			quaternion& slerp(quaternion q1, quaternion q2,
					f32 time, f32 threshold=.05f);

			//! Create quaternion from rotation angle and rotation axis.
			/** Axis must be unit length.
			The quaternion representing the rotation is
			q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k).
			\param angle Rotation Angle in radians.
			\param axis Rotation axis. */
			quaternion& fromAngleAxis (f32 angle, const Vector3DF& axis);

			//! Fills an angle (radians) around an axis (unit vector)
			void toAngleAxis (f32 &angle, Vector3DF& axis) const;

			f64 getAngle ()			{ return (W>1.0) ? 0 : 2.0f * acosf(W) * RADtoDEG; }			

			//! Output this quaternion to an euler angle (radians)
			void toEuler(Vector3DF& euler) const;

			//! Rotate a vector by this quaternion
			Vector3DF rotateVec(Vector3DF v);

			//! Set quaternion to identity
			quaternion& makeIdentity();

			//! Set quaternion to represent a rotation from one vector to another.
			void rotationFromTo(const Vector3DF& from, const Vector3DF& to);

			//! Quaternion elements.
			f64 X; // vectorial (imaginary) part
			f64 Y;
			f64 Z;
			f64 W; // real part
	};


	// Constructor which converts euler angles to a quaternion
	inline quaternion::quaternion(f32 x, f32 y, f32 z)
	{
		set(x,y,z);
	}


	// Constructor which converts euler angles to a quaternion
	inline quaternion::quaternion(const Vector3DF& vec)
	{
		set(vec.x,vec.y,vec.z);
	}

	// Constructor which converts a matrix to a quaternion
	inline quaternion::quaternion(const Matrix4F& mat)
	{
		(*this) = mat;
	}

	// equal operator
	inline bool quaternion::operator==(const quaternion& other) const
	{
		return ((X == other.X) &&
			(Y == other.Y) &&
			(Z == other.Z) &&
			(W == other.W));
	}

	// inequality operator
	inline bool quaternion::operator!=(const quaternion& other) const
	{
		return !(*this == other);
	}

	// assignment operator
	inline quaternion& quaternion::operator=(const quaternion& other)
	{
		X = other.X;
		Y = other.Y;
		Z = other.Z;
		W = other.W;
		return *this;
	}

	// matrix assignment operator
	inline quaternion& quaternion::operator=(const Matrix4F& m)
	{
		const f32 diag = m.data[0] + m.data[5] + m.data[10] + 1;

		if( diag > 0.0f )
		{
			const f32 scale = sqrtf(diag) * 2.0f; // get scale from diagonal

			// TODO: speed this up
			X = (m.data[6] - m.data[9]) / scale;
			Y = (m.data[8] - m.data[2]) / scale;
			Z = (m.data[1] - m.data[4]) / scale;
			W = 0.25f * scale;
		}
		else
		{
			if (m.data[0]>m.data[5] && m.data[0]>m.data[10])
			{
				// 1st element of diag is greatest value
				// find scale according to 1st element, and double it
				const f32 scale = sqrtf(1.0f + m.data[0] - m.data[5] - m.data[10]) * 2.0f;

				// TODO: speed this up
				X = 0.25f * scale;
				Y = (m.data[4] + m.data[1]) / scale;
				Z = (m.data[2] + m.data[8]) / scale;
				W = (m.data[6] - m.data[9]) / scale;
			}
			else if (m.data[5]>m.data[10])
			{
				// 2nd element of diag is greatest value
				// find scale according to 2nd element, and double it
				const f32 scale = sqrtf(1.0f + m.data[5] - m.data[0] - m.data[10]) * 2.0f;

				// TODO: speed this up
				X = (m.data[4] + m.data[1]) / scale;
				Y = 0.25f * scale;
				Z = (m.data[9] + m.data[6]) / scale;
				W = (m.data[8] - m.data[2]) / scale;
			}
			else
			{
				// 3rd element of diag is greatest value
				// find scale according to 3rd element, and double it
				const f32 scale = sqrtf(1.0f + m.data[10] - m.data[0] - m.data[5]) * 2.0f;

				// TODO: speed this up
				X = (m.data[8] + m.data[2]) / scale;
				Y = (m.data[9] + m.data[6]) / scale;
				Z = 0.25f * scale;
				W = (m.data[1] - m.data[4]) / scale;
			}
		}

		return normalize();
	}


	// multiplication operator
	inline quaternion quaternion::operator*(const quaternion& other) const
	{
		quaternion tmp;

		/*tmp.W = (other.W * W) - (other.X * X) - (other.Y * Y) - (other.Z * Z);
		tmp.X = (other.W * X) + (other.X * W) + (other.Y * Z) - (other.Z * Y);
		tmp.Y = (other.W * Y) + (other.Y * W) + (other.Z * X) - (other.X * Z);
		tmp.Z = (other.W * Z) + (other.Z * W) + (other.X * Y) - (other.Y * X);*/

		tmp.W = (W * other.W) - (X * other.X) - (Y * other.Y) - (Z * other.Z);
		tmp.X = (W * other.X) + (X * other.W) + (Y * other.Z) - (Z * other.Y);
		tmp.Y = (W * other.Y) + (Y * other.W) + (Z * other.X) - (X * other.Z);
		tmp.Z = (W * other.Z) + (Z * other.W) + (X * other.Y) - (Y * other.X);

		return tmp;
	}


	// multiplication operator
	inline quaternion quaternion::operator*(f32 s) const
	{
		return quaternion(s*X, s*Y, s*Z, s*W);
	}


	// multiplication operator
	inline quaternion& quaternion::operator*=(f32 s)
	{
		X*=s;
		Y*=s;
		Z*=s;
		W*=s;
		return *this;
	}

	// multiplication operator
	inline quaternion& quaternion::operator*=(const quaternion& other)
	{
		return (*this = other * (*this));
	}

	// add operator
	inline quaternion quaternion::operator+(const quaternion& b) const
	{
		return quaternion(X+b.X, Y+b.Y, Z+b.Z, W+b.W);
	}

	// Creates a matrix from this quaternion
	inline Matrix4F quaternion::getMatrix() const
	{
		Matrix4F m;
		getMatrix(m);
		return m;
	}

	/*!
		Creates a matrix from this quaternion
	*/
	inline void quaternion::getMatrix(Matrix4F &dest,
			const Vector3DF &center) const
	{
		dest.data[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
		dest.data[1] = 2.0f*X*Y + 2.0f*Z*W;
		dest.data[2] = 2.0f*X*Z - 2.0f*Y*W;
		dest.data[3] = 0.0f;

		dest.data[4] = 2.0f*X*Y - 2.0f*Z*W;
		dest.data[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
		dest.data[6] = 2.0f*Z*Y + 2.0f*X*W;
		dest.data[7] = 0.0f;

		dest.data[8] = 2.0f*X*Z + 2.0f*Y*W;
		dest.data[9] = 2.0f*Z*Y - 2.0f*X*W;
		dest.data[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
		dest.data[11] = 0.0f;

		dest.data[12] = center.x;
		dest.data[13] = center.y;
		dest.data[14] = center.z;
		dest.data[15] = 1.f;

		//dest.Identity();
	}


	/*!
		Creates a matrix from this quaternion
		Rotate about a center point
		shortcut for
		core::quaternion q;
		q.rotationFromTo(vin[i].Normal, forward);
		q.getMatrix(lookat, center);

		core::Matrix4F m2;
		m2.setInverseTranslation(center);
		lookat *= m2;
	*/
	inline void quaternion::getMatrixCenter(Matrix4F &dest,
						const Vector3DF &center,
						const Vector3DF &translation) const
	{
		dest.data[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
		dest.data[1] = 2.0f*X*Y + 2.0f*Z*W;
		dest.data[2] = 2.0f*X*Z - 2.0f*Y*W;
		dest.data[3] = 0.0f;

		dest.data[4] = 2.0f*X*Y - 2.0f*Z*W;
		dest.data[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
		dest.data[6] = 2.0f*Z*Y + 2.0f*X*W;
		dest.data[7] = 0.0f;

		dest.data[8] = 2.0f*X*Z + 2.0f*Y*W;
		dest.data[9] = 2.0f*Z*Y - 2.0f*X*W;
		dest.data[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
		dest.data[11] = 0.0f;

		//dest.setRotationCenter ( center, translation );
	}

	// Creates a matrix from this quaternion
	inline void quaternion::getMatrix_transposed(Matrix4F &dest) const
	{
		dest.data[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
		dest.data[4] = 2.0f*X*Y + 2.0f*Z*W;
		dest.data[8] = 2.0f*X*Z - 2.0f*Y*W;
		dest.data[12] = 0.0f;

		dest.data[1] = 2.0f*X*Y - 2.0f*Z*W;
		dest.data[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
		dest.data[9] = 2.0f*Z*Y + 2.0f*X*W;
		dest.data[13] = 0.0f;

		dest.data[2] = 2.0f*X*Z + 2.0f*Y*W;
		dest.data[6] = 2.0f*Z*Y - 2.0f*X*W;
		dest.data[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
		dest.data[14] = 0.0f;

		dest.data[3] = 0.f;
		dest.data[7] = 0.f;
		dest.data[11] = 0.f;
		dest.data[15] = 1.f;

		//dest.setDefinitelyIdentityMatrix(false);
	}


	// Inverts this quaternion
	inline quaternion& quaternion::makeInverse()
	{
		X = -X; Y = -Y; Z = -Z;
		return *this;
	}


	// sets new quaternion
	inline quaternion& quaternion::set(f32 x, f32 y, f32 z, f32 w)
	{
		X = x;
		Y = y;
		Z = z;
		W = w;
		return *this;
	}

	// sets new quaternion based on euler angles
	inline quaternion& quaternion::set(f32 x, f32 y, f32 z)
	{
		f64 angle;

		angle = x * 0.5;
		const f64 sr = sin(angle);
		const f64 cr = cos(angle);

		angle = y * 0.5;
		const f64 sp = sin(angle);
		const f64 cp = cos(angle);

		angle = z * 0.5;
		const f64 sy = sin(angle);
		const f64 cy = cos(angle);

		const f64 cpcy = cp * cy;
		const f64 spcy = sp * cy;
		const f64 cpsy = cp * sy;
		const f64 spsy = sp * sy;

		X = (f32)(sr * cpcy - cr * spsy);
		Y = (f32)(cr * spcy + sr * cpsy);
		Z = (f32)(cr * cpsy - sr * spcy);
		W = (f32)(cr * cpcy + sr * spsy);

		return normalize();
	}

	// sets new quaternion based on euler angles
	inline quaternion& quaternion::set(const Vector3DF& vec)
	{
		return set(vec.x*DEGtoRAD,vec.y*DEGtoRAD,vec.z*DEGtoRAD);
	}

	// sets new quaternion based on other quaternion
	inline quaternion& quaternion::set(const quaternion& quat)
	{
		return (*this=quat);
	}


	//! returns if this quaternion equals the other one, taking floating point rounding errors into account
	inline bool quaternion::equals(const quaternion& other, const f32 tolerance) const
	{
		return fequal(X, other.X, tolerance) &&
			fequal(Y, other.Y, tolerance) &&
			fequal(Z, other.Z, tolerance) &&
			fequal(W, other.W, tolerance);
	}

	inline float reciprocal_squareroot ( float number )
	{
	  long i;
	  float x2, y;
	  const float threehalfs = 1.5F;

	  x2 = number * 0.5F;
	  y  = number;
	  i  = * ( long * ) &y;
	  i  = 0x5f3759df - ( i >> 1 );
	  y  = * ( float * ) &i;
	  y  = y * ( threehalfs - ( x2 * y * y ) );

	  #ifndef Q3_VM
	  #ifdef __linux__
		assert( !isnan(y) );
	  #endif
	  #endif
	  return y;
	}

	// normalizes the quaternion
	inline quaternion& quaternion::normalize()
	{
		const f32 n = X*X + Y*Y + Z*Z + W*W;

		if (n == 1)
			return *this;

		//n = 1.0f / sqrtf(n);
		return (*this *= reciprocal_squareroot ( n ));
	}


	// set this quaternion to the result of the linear interpolation between two quaternions
	inline quaternion& quaternion::lerp(quaternion q1, quaternion q2, f32 time)
	{
		const f32 scale = 1.0f - time;
		return (*this = (q1*scale) + (q2*time));
	}


	// set this quaternion to the result of the interpolation between two quaternions
	inline quaternion& quaternion::slerp(quaternion q1, quaternion q2, f32 time, f32 threshold)
	{
		f32 angle = q1.dotProduct(q2);

		// make sure we use the short rotation
		if (angle < 0.0f)
		{
			q1 *= -1.0f;
			angle *= -1.0f;
		}

		if (angle <= (1-threshold)) // spherical interpolation
		{
			const f32 theta = acosf(angle);
			const f32 invsintheta = 1.0f / sinf(theta);
			const f32 scale = sinf(theta * (1.0f-time)) * invsintheta;
			const f32 invscale = sinf(theta * time) * invsintheta;
			return (*this = (q1*scale) + (q2*invscale));
		}
		else // linear interploation
			return lerp(q1,q2,time);
	}


	// calculates the dot product
	inline f32 quaternion::dotProduct(const quaternion& q2) const
	{
		return (X * q2.X) + (Y * q2.Y) + (Z * q2.Z) + (W * q2.W);
	}


	//! axis must be unit length, angle in radians
	inline quaternion& quaternion::fromAngleAxis(f32 angle, const Vector3DF& axis)
	{
		angle *= 0.5f;
		const f32 fSin = sinf( angle );
		W = cosf( angle );
		X = fSin * axis.x;
		Y = fSin * axis.y;
		Z = fSin * axis.z;
		return *this;
	}


	inline void quaternion::toAngleAxis(f32 &angle, Vector3DF &axis) const
	{
		const f32 scale = sqrtf(X*X + Y*Y + Z*Z);

		if (fabs(scale)<EPS || W > 1.0f || W < -1.0f)
		{
			angle = 0.0f;
			axis.x = 0.0f;
			axis.y = 1.0f;
			axis.z = 0.0f;
		}
		else
		{
			const f32 invscale = 1.0 / scale;
			angle = 2.0f * acosf(W);
			axis.x = X * invscale;
			axis.y = Y * invscale;
			axis.z = Z * invscale;
		}
	}

	inline void quaternion::toEuler(Vector3DF& euler) const
	{
		const f64 test = (X*Y + Z*W);

		if ( fequal (test, 0.5, 0.000001))
		{
			euler.x = 0;							// roll = rotation about x-axis (bank)
			euler.y = (f32)(PI64 / 2.0);			// pitch = rotation about y-axis (attitude)
			euler.z = (f32) (-2.0*atan2(X, W));		// yaw = rotation about z-axis (heading)		
		}
		else if ( fequal (test, -0.5, 0.000001))
		{
			// heading = rotation about z-axis
			euler.z = (f32) (2.0*atan2(X, W));
			// bank = rotation about x-axis
			euler.x = 0;
			// attitude = rotation about y-axis
			euler.y = (f32) (PI64/-2.0);
		}
		else 
		{			
			euler.x = (f32) atan2(2.0 * (X*W - Y*Z), 1.0-2.0f*(X*X+Z*Z) );	// roll = rotation about x-axis (bank)
			euler.y = (f32) asin( 2.0*test );								// pitch = rotation about y-axis (attitude)
			euler.z = (f32) atan2(2.0 * (X*Z - Y*W), 1.0-2.0f*(Y*Y+Z*Z) );	// yaw = rotation about z-axis (heading)
		}

		euler.x *= RADtoDEG;
		euler.y *= RADtoDEG;
		euler.z *= RADtoDEG;
	}


	inline Vector3DF quaternion::operator* (const Vector3DF& v) const
	{
		// nVidia SDK implementation

		Vector3DF uv, uuv;
		Vector3DF qvec; qvec.Set( float(X), float(Y), float(Z) );
		uv = qvec.Cross (v);
		uuv = qvec.Cross (uv);
		uv *= (2.0f * float(W) );
		uuv *= 2.0f;

		uuv += uv;
		uuv += v;
		return uuv;
	}

	// set quaternion to identity
	inline quaternion& quaternion::makeIdentity()
	{
		W = 1.f;
		X = 0.f;
		Y = 0.f;
		Z = 0.f;
		return *this;
	}

	inline Vector3DF quaternion::rotateVec(Vector3DF p)
	{
		//--- efficient way		
		Vector3DF u;
		u.Set ( float(X), float(Y), float(Z) );		
		Vector3DF q;				
		q = u * float(2.0f*u.Dot(p)) + p * float(W*W-u.Dot(u)) + Vector3DF::Cross(u, p) * float(2.0f*W);			// Cross(u,p) - u and p preserved
		
		/*--- full quaterion-to-matrix solution (correct)
		Vector3DF q;
		q.x = p.x*(1.0f - 2.0f*Y*Y - 2.0f*Z*Z)		+ p.y*(2.0f*X*Y - 2.0f*Z*W)			+ p.z*(2.0f*X*Z + 2.0f*Y*W);
		q.y = p.x*(2.0f*X*Y + 2.0f*Z*W)				+ p.y*(1.0f - 2.0f*X*X - 2.0f*Z*Z)	+ p.z*(2.0f*Z*Y - 2.0f*X*W);
		q.z = p.x*(2.0f*X*Z - 2.0f*Y*W)				+ p.y*(2.0f*Z*Y + 2.0f*X*W)			+ p.z*(1.0f - 2.0f*X*X - 2.0f*Y*Y); */

		return q;
	}

	
		
		/*dest.data[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
		dest.data[1] = 2.0f*X*Y + 2.0f*Z*W;
		dest.data[2] = 2.0f*X*Z - 2.0f*Y*W;
		dest.data[3] = 0.0f;

		dest.data[4] = 2.0f*X*Y - 2.0f*Z*W;
		dest.data[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
		dest.data[6] = 2.0f*Z*Y + 2.0f*X*W;
		dest.data[7] = 0.0f;

		dest.data[8] = 2.0f*X*Z + 2.0f*Y*W;
		dest.data[9] = 2.0f*Z*Y - 2.0f*X*W;
		dest.data[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
		dest.data[11] = 0.0f;

		dest.data[12] = center.x;
		dest.data[13] = center.y;
		dest.data[14] = center.z;
		dest.data[15] = 1.f; */



	inline void quaternion::rotationFromTo(const Vector3DF& from, const Vector3DF& to)
	{
		// Based on Stan Melax's article in Game Programming Gems
		// Copy, since cannot modify local
		/*Vector3DF v0 = from;
		Vector3DF v1 = to;
		v0.Normalize();
		v1.Normalize();

		const f32 d = v0.Dot (v1);
		if (d >= 1.0f) // If dot == 1, vectors are the same
		{
			return makeIdentity();
		}
		else if (d <= -1.0f) // exactly opposite
		{
			Vector3DF axis(1.0f, 0.f, 0.f);
			axis = axis.Cross(v0);
			if (axis.Length()==0)
			{
				axis.Set(0.f,1.f,0.f);
				axis.Cross(v0);
			}
			// same as fromAngleAxis(core::PI, axis).normalize();
			return set(axis.x, axis.y, axis.z, 0).normalize();
		}

		const f32 s = sqrtf( (1+d)*2 ); // optimize inv_sqrt
		const f32 invs = 1.f / s;
		const Vector3DF c = v0.Cross(v1)*invs;
		return set(c.x, c.y, c.z, s * 0.5f).normalize();*/

		Vector3DF v0 = from;
		Vector3DF v1 = to;
		Vector3DF v2 = from;
		v2.Cross(v1);
		v2.Normalize();							// cross product = axis of rotation, cross of 'from' and 'to' vectors 
		v0.Normalize(); 
		v1.Normalize();
		fromAngleAxis (acos(v0.Dot(v1)), v2);	// dot product = angle of rotation between 'from' and 'to' vectors
		normalize();
	}

#endif