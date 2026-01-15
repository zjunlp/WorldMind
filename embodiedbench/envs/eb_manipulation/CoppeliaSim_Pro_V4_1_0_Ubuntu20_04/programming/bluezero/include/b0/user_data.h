#ifndef B0__USER_DATA_H__INCLUDED
#define B0__USER_DATA_H__INCLUDED

namespace b0
{

//! \cond HIDDEN_SYMBOLS

class UserData
{
public:
    inline void setUserData(void *user_data)
    {
        user_data_ = user_data;
    }

    inline void * getUserData() const
    {
        return user_data_;
    }

private:
    void *user_data_;
};

//! \endcond

} // namespace b0

#endif // B0__USER_DATA_H__INCLUDED
