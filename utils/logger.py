"""
Function Name: logger.py
*
Description: Create a logger which can fully replace Python built-in function "print"
             See logging.setLoggerClass to design a customized logger
*
Argument: None
*
Parameters: None
*
Return: None
*
Edited by: [2020-10-22] [Bill Gao]
           [2020-11-19] [Bill Gao]
           - Add new function getShowTime
"""
import logging
import sys
import os
import csv
import traceback
import inspect
from errno import EACCES
from time import strftime
class LevelFilter(logging.Filter):    
    def __init__(self, filter_name, default_level, except_level):
        """
        Function Name: __init__
        *
        Description: If Handler.addFilter is set, every raised log whose level is higher than Handler.setlevel 
                     will be passed through filter. 
        *
        Argument: None
        *
        Parameters: 
                    default_level [int] -> should be Handler.setlevel
                    except_level  [int] -> this level should be higher than Handler.setlevel
                                           and would be ignore by LevelFilter
        *
        Return: None
        *
        Edited by: [2020-10-22] [Bill Gao]
        """        
        super().__init__(filter_name)
        self.default_level = default_level
        self.except_level = except_level

    def filter(self, record):
        """
        Function Name: filter
        *
        Description: Filters are consulted in turn by applied logger, if none of them
                     return false, the record will be processed. If one returns false, then
                     no further processing occurs.
                     This is a Built_in function of logging.Filter, do not change the name of this
                     subroutine.
        *
        Argument: None
        *
        Parameters: 
                    record [logging.LogRecord] -> message be passed through logger
        *
        Return: [bool] -> True if level of raised log is higher than default_level
                          False if level of raised log equals to except_level
        *
        Edited by: [2020-10-22] [Bill Gao]
        """        
        if record.levelno==self.except_level:
            return False
        elif record.levelno>=self.default_level:
            return True

class Log:
    def __init__(self, name:str, stream_level="DEBUG", stream_enable=True, record_level="ERROR", record_path='log.csv'):
        """
        Function Name: __init__
        *
        Description: Log is created to replace the built-in function print in Python. Not only
                     it provides very useful information including when does the log happen,
                     happening in which file, or even happening in while subroutine in the file.
                     Also, it provides a extremely practical level system listed in below and
                     are weighted from left to right.
                                    [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                                    [   10,   20,      30,    40,       50]
                     You can control which log to print and record by setting the stream_level 
                     and the record_level.
        *
        Argument: None
        *
        Parameters: 
                    name [str] -> name of the logger
                    stream_level [str] -> should be one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                    record_level [str] -> should be one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                    record_path [str] -> record path. Ex: "setting/log.csv"
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        # verify type of name
        assert name != str, "name should be str!"

        # check whether csv file exists or is opening
        self.__csvCreater(record_path)

        # setting
        self.__logger = logging.Logger(str(name), level=logging.DEBUG)
        self.__record_enable = bool(record_path)
        self.__stream_enable = bool(stream_enable)
        self.__record_only_level = 60    
        self.__stream_only_level = 70
        self.__stream_default_level = self.__levelCheck(stream_level)
        self.__record_default_level = self.__levelCheck(record_level)
        self.__show_time = "Empty"


        # stream console setting
        if self.__stream_enable:
            stream_filter = LevelFilter(filter_name = "stream",
                                    default_level=self.__stream_default_level,
                                    except_level =self.__record_only_level)
            stream__formatter = logging.Formatter("\n[%(show_time)s], [%(filename)s], [%(error_file)s.%(error_func)s], [%(level_name)s], \n%(message)s")
            stream_console = logging.StreamHandler()
            stream_console.addFilter(stream_filter)
            stream_console.setFormatter(stream__formatter)
            self.__logger.addHandler(stream_console)

        # record console setting
        if self.__record_enable:
            record_filter = LevelFilter(filter_name = "record",
                                    default_level=self.__record_default_level,
                                    except_level =self.__stream_only_level)
            record_formatter = logging.Formatter("[%(show_time)s], [%(filename)s], [%(error_file)s.%(error_func)s], [%(level_name)s], %(message)s")
            record_console = logging.FileHandler(record_path)
            record_console.addFilter(record_filter)
            record_console.setFormatter(record_formatter)
            self.__logger.addHandler(record_console)
            


    def __csvCreater(self, record_path):
        """
        Function Name: __csvCreater
        *
        Description: create csv file at record_path if it doesn't exist, and
                     check whether it is opening
        *
        Argument: None
        *
        Parameters: 
                    record_path [str] -> The path where the file is 
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        
        # create a csv file with header if it doesn't exist at the record_path
        log_file_path = os.path.abspath(record_path)
        if os.path.exists(log_file_path) == False:
            with open(record_path,'w', newline="") as file:
                header = ["Time", "Logging_File", "Function_Name", "Error_Level", "Error_Message"]
                writedCsv = csv.writer(file)
                writedCsv.writerow(header)  

        # check if csv file is opening or not
        try:
            open_check = open(record_path, "r+") # or "a+", whatever you need
            open_check.close()
        except IOError as x:
            if x.errno == EACCES:
                error_message = f"File: {record_path} can't be read, please close the file."
                raise IOError(error_message)
            else:
                error_message = f"File: {record_path} can't be read."
                raise IOError(error_message)

    def __levelCheck(self, level):
        """
        Function Name: __levelCheck
        *
        Description: check whether input level fits the format and convert
                     its cooresponding integer
                     [DEBUG,INFO,WARNING,ERROR,CRITICAL] -> [10,20,30,40,50]
        *
        Argument: None
        *
        Parameters: 
                    level [str] -> level of input log      
        *
        Return: 
                [int] -> level of input log
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        # verify the content of level
        if level.upper() == "DEBUG":
            output = logging.DEBUG
        elif level.upper() == "INFO":
            output = logging.INFO
        elif level.upper() == "WARNING":
            output = logging.WARNING
        elif level.upper() == "ERROR":
            output = logging.ERROR
        elif level.upper() == "CRITICAL":
            output = logging.CRITICAL
        else:
            raise ValueError(
                "loglevel should be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        return output      
    
    def getShowTime(self):
        """
        Function Name: getShowTime
        *
        Description: This will return the time of the latest calling of function show 
        *
        Argument: None
        *
        Parameters: None
        *
        Return: 
                [str] -> Return the time of the latest calling of function show, or
                         return "Enpty" if there has no recent calling
        *
        Edited by: [2020-11-19] [Bill Gao]
        """        
        return self.__show_time
        
    def show(self, input_message, log_level):
        """
        Function Name: show
        *
        Description: show the input message
        *
        Argument: None
        *
        Parameters: 
                    input_message [str] -> message you want to show
                    log_level [str] -> should be one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        *
        Return: None
        *
        Edited by: [2020-10-23] [Bill Gao]
        """        
        # verify the content of record_level
        log_level = self.__levelCheck(log_level)
        self.__show_time = strftime("%Y-%m-%d %H:%M:%S")
        
        if log_level >= self.__stream_default_level or log_level >= self.__record_default_level:
            # get the name of input log_level
            error_dict = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
            for key, value in error_dict.items():
                if value == log_level:
                    self.__except_level_name = key
            logging.addLevelName(self.__record_only_level, self.__except_level_name)

            # Get the defail from the input_message including error_file, error_func, etc if the input_message
            # is an error_message, otherwise just print it out
            if input_message.__class__ == str:
                extra_message = {"show_time": self.__show_time,
                                 "error_file": inspect.stack()[1][1].split("\\")[-1].split(".")[0],
                                 "error_func": inspect.stack()[1][3],
                                 "level_name": self.__except_level_name}
                output_message = input_message
            else:
                detail = input_message.args[0] if any(input_message.args) == True else "Error message is empty"
                error_type = input_message.__class__.__name__
                cl, exc, tb = sys.exc_info()
                last_call_stack = traceback.extract_tb(tb)[-1]
                error_file = last_call_stack[0]
                line_num = last_call_stack[1]
                error_func = last_call_stack[2]
                extra_message = {"show_time": self.__show_time,
                                 "error_file": error_file.split("\\")[-1].split(".")[0],
                                 "error_func": error_func,
                                 "level_name": self.__except_level_name}
                output_message = "File \"{}\", line {}, in {}: [{}] {}".format(error_file, line_num, error_func, error_type, detail)

            # show the original output_message and record the revised output_message in .csv file which replaces "," with "-" 
            # to keep the "open file in editor" function in VS Code, also let it suit the format of .csv file
            if output_message.find(","):
                revised_message = output_message.replace(",", " -")
                if log_level >= self.__stream_default_level:
                    self.__logger.log(self.__stream_only_level, output_message, extra=extra_message)
                if log_level >= self.__record_default_level and self.__record_enable:
                    self.__logger.log(self.__record_only_level, revised_message, extra=extra_message)
            else:
                if log_level >= self.__stream_default_level:
                    self.__logger.log(self.__stream_only_level, output_message, extra=extra_message)
                if log_level >= self.__record_default_level and self.__record_enable:
                    self.__logger.log(self.__record_only_level, output_message, extra=extra_message)