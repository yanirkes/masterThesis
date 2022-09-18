import psycopg2
from configparser import ConfigParser
from sqlalchemy import create_engine
import pandas as pd


class PostgressCon():
    def __init__(self, ):
        self.conn, self.cur = self.connect()

    @classmethod
    def config(cls, filename=r"C:\Users\yanir\PycharmProjects\masterThesis\utils\database.ini", section='postgresql'):
        # create a parser
        parser = ConfigParser()

        # read config file
        parser.read(filename)

        # get section, default to postgresql
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception("section {0} not found in the {1} file".format(section, filename))
        print(db)
        return db

    def connect(self):
        """ Connect to the PostgreSQL database server """
        try:
            # read connection parameters
            params = self.config()

            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params)
            print(conn)
            if conn is not None:
                self.conn = conn
            else:
                raise Exception("conn is None")

            # create a cursor
            return conn, conn.cursor()

        except (Exception, psycopg2.DatabaseError) as error:
            if self.conn is not None:
                self.conn.close()
            print(error)

        except psycopg2.Error as e:
            print(e)
            self.conn.close()

    def execute_query(self, q):
        try:
            # execute a statement
            print('RUN QUERY')
            self.cur.execute(q)

            print('FETCH COLUMNS HEADER')
            column_names = [header[0] for header in self.cur.description]

            print('FETCH RESULTS')
            results = self.cur.fetchall()
            return results, column_names

        except psycopg2.Error as e:
            print(e)
            self.conn.close()

    def execute_query_with_headers(self, q):
        try:
            # execute a statement
            print('RUN QUERY')
            self.cur.execute(q)

            print('FETCH COLUMNS HEADER')
            column_names = [header[0] for header in self.cur.description]

            print('FETCH RESULTS')
            results = self.cur.fetchall()
            return results, column_names

        except psycopg2.Error as e:
            print(e)
            self.conn.close()

    def close_connection(self):
        self.conn.close()
        print("Is the connection is closed?: ", self.conn.closed == 1)

    def table_from_df(self, data, schema, table):
        config_copy = self.config()
        url = "postgresql://" + config_copy['user'] + ":" + str(
            config_copy['password']) + "@" + str(config_copy['host']) + "/" + str(
            config_copy['database'])

        engine = create_engine(url)
        table_mame = schema +"."+ table
        data.to_sql(name= table, schema = schema, con = engine)
        print("DONE CREATING TABLE %s", table_mame)


# if __name__ == '__main__':
#     dbConn_obj = PostgressCon()
#
#
#     temp = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 3]})
#     dbConn_obj.table_from_df(temp, 'basket', 'table_to_delete')
#
#     q = """select *
#             from basket.initial_db"""
#     result = dbConn_obj.execute_query(q)
#     print(result)
#
#
#     for i in temp.items():
#         print(i[0])
