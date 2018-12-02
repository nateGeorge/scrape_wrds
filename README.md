# scrape_wrds
getting data from WRDS for stocks


# fixing pgpass/password issues
To avoid logging in every time, you can set pgpass, but it wasn't working for me (something about logged in username needs to match db username on Linux).

Instead, I added a kwarg for password, and used it in the postgres login:
/usr/local/lib/python3.6/dist-packages/wrds
added the line in init, and commented out the `self._password = ""`:
`self._password = kwargs.get('wrds_password', None)`
